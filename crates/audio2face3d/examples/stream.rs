//! Stream raw mono signed PCM16 little-endian input; no executor dependency in direct mode.
use audio2face3d::client::{Client, types::*};
use std::{
    future::Future,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
    time::Duration,
};

struct Signal(std::thread::Thread);
impl Wake for Signal {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}
fn wait<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(Signal(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        if let Poll::Ready(result) = future.as_mut().poll(&mut context) {
            return result;
        }
        std::thread::park();
    }
}

// Both modes use this function unchanged. A separate sender prevents bounded-queue deadlocks.
fn stream(client: &Client, bytes: Vec<u8>) -> Result<()> {
    let options = RequestOptions::builder(AudioFormat::MONO_16KHZ)
        .timeout(Duration::from_secs(120))
        .build()?;
    let (mut input, mut output, control) = client.start(options)?.split();
    std::thread::scope(|scope| {
        let sender = scope.spawn(move || -> Result<()> {
            for chunk in bytes.chunks(3200) {
                wait(input.send(InputChunk::new(
                    PcmBuffer::from_vec(chunk.to_vec())?,
                    vec![],
                )))?;
            }
            wait(input.finish())
        });
        let received = (|| -> Result<()> {
            let mut completed = false;
            let mut frames = 0;
            while let Some(event) = wait(output.recv())? {
                match event {
                    OutputEvent::Curves(_) => frames += 1,
                    OutputEvent::Completed(_) => completed = true,
                    _ => {}
                }
            }
            if !completed {
                return Err(Error::new(
                    ErrorKind::IncompleteResponse,
                    "missing completion",
                ));
            }
            println!("Completed: {frames} curve frames");
            Ok(())
        })();
        if received.is_err() {
            control.cancel();
        }
        let sent = sender
            .join()
            .map_err(|_| Error::new(ErrorKind::Inference, "sender panicked"))
            .and_then(|r| r);
        let closed = wait(control.closed());
        received.and(sent).and(closed)
    })
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let mode = args
        .get(1)
        .ok_or("usage: stream direct MODEL INPUT.pcm | server URL INPUT.pcm | mock INPUT.pcm")?;
    let input_path = args.last().ok_or("input path required")?;
    let bytes = std::fs::read(input_path)?;
    #[cfg(feature = "client-grpc")]
    let runtime;
    let client = match mode.as_str() {
        #[cfg(feature = "native")]
        "direct" => wait(Client::direct(
            audio2face3d::client::DirectConfig::builder(
                audio2face3d::client::InferenceConfig::builder(
                    audio2face3d::client::BackendKind::Regression,
                )
                .model(args.get(2).ok_or("model required")?)
                .build()?,
            )
            .build()?,
        ))?,
        #[cfg(feature = "mock")]
        "mock" => wait(Client::direct(
            audio2face3d::client::DirectConfig::builder(
                audio2face3d::client::InferenceConfig::builder(
                    audio2face3d::client::BackendKind::Mock,
                )
                .build()?,
            )
            .build()?,
        ))?,
        #[cfg(feature = "client-grpc")]
        "server" => {
            // Only server initialization creates Tokio. It outlives client shutdown below.
            runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            let config =
                audio2face3d::client::ServerConfig::builder(args.get(2).ok_or("URL required")?)
                    .runtime(runtime.handle().clone())
                    .build()?;
            wait(Client::server(config))?
        }
        _ => return Err("unknown mode or required feature not enabled".into()),
    };
    let result = stream(&client, bytes);
    let shutdown = wait(client.shutdown());
    result?;
    shutdown?;
    Ok(())
}
