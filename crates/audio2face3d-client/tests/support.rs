#![allow(dead_code)]
use audio2face3d_client::{Client, types::*};
use std::{
    future::Future,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
    time::Duration,
};
struct Wakeup(std::sync::mpsc::Sender<()>);
impl Wake for Wakeup {
    fn wake(self: Arc<Self>) {
        let _ = self.0.send(());
    }
}
pub fn wait<F: Future>(future: F) -> F::Output {
    let (tx, rx) = std::sync::mpsc::channel();
    let w = Waker::from(Arc::new(Wakeup(tx)));
    let mut cx = Context::from_waker(&w);
    let mut f = std::pin::pin!(future);
    loop {
        if let Poll::Ready(r) = f.as_mut().poll(&mut cx) {
            return r;
        }
        rx.recv_timeout(Duration::from_secs(180))
            .expect("future did not wake");
    }
}
pub fn pcm(seed: i16, frames: usize) -> Vec<u8> {
    (0..frames)
        .flat_map(|i| seed.wrapping_add((i % 200) as i16).to_le_bytes())
        .collect()
}
pub fn collect(client: &Client, options: RequestOptions, pcm: Vec<u8>) -> Result<Vec<OutputEvent>> {
    let (mut input, mut output, control) = client.start(options)?.split();
    std::thread::scope(|scope| {
        let sending = scope.spawn(move || {
            for part in pcm.chunks(1066) {
                wait(input.send(InputChunk::new(PcmBuffer::from_vec(part.to_vec())?, vec![])))?;
            }
            wait(input.finish())
        });
        let result = (|| {
            let mut events = vec![];
            while let Some(event) = wait(output.recv())? {
                events.push(event);
            }
            Ok(events)
        })();
        if result.is_err() {
            control.cancel();
        }
        let sent = sending.join().unwrap();
        let closed = wait(control.closed());
        result.and_then(|events| {
            sent?;
            closed?;
            assert!(matches!(events.last(), Some(OutputEvent::Completed(_))));
            Ok(events)
        })
    })
}
pub fn returned_pcm(events: &[OutputEvent]) -> Vec<u8> {
    events
        .iter()
        .filter_map(|e| {
            if let OutputEvent::Audio(a) = e {
                Some(a.pcm().as_bytes())
            } else {
                None
            }
        })
        .flatten()
        .copied()
        .collect()
}
