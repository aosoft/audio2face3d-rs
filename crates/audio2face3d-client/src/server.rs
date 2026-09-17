use crate::{Client, Limits, core::Lease, driver::*, notify::Subscription, types::*};
use audio2face3d_protocol::{
    convert,
    wire::{
        controller,
        nvidia_ace::services::a2f_controller::v1::a2f_controller_service_client::A2fControllerServiceClient,
    },
};
use std::{
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
    time::Duration,
};
use tokio::runtime::Handle;
use tokio_stream::Stream;
use tonic::transport::{Channel, Endpoint};

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub endpoint: String,
    /// None selects try_current during initialization; no runtime is created.
    pub runtime: Option<Handle>,
    pub limits: Limits,
    pub connect_timeout: Duration,
    pub max_message_bytes: usize,
    /// Fixed HTTP/2 connection and stream receive windows; adaptive growth is disabled.
    pub http2_window_bytes: u32,
}
impl ServerConfig {
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            runtime: None,
            limits: Limits::default(),
            connect_timeout: Duration::from_secs(10),
            max_message_bytes: 4 * 1024 * 1024,
            http2_window_bytes: 65_535,
        }
    }
}
struct Server {
    handle: Handle,
    channel: Mutex<Option<Channel>>,
    max_message_bytes: usize,
}
impl Client {
    pub async fn server(config: ServerConfig) -> Result<Self> {
        config.limits.validate()?;
        if config.connect_timeout.is_zero()
            || config.max_message_bytes == 0
            || !(65_535..=0x7fff_ffff).contains(&config.http2_window_bytes)
        {
            return Err(Error::invalid("invalid server transport limits"));
        }
        let handle = config
            .runtime
            .or_else(|| Handle::try_current().ok())
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::RuntimeUnavailable,
                    "server mode requires a driven Tokio runtime with I/O and time enabled",
                )
            })?;
        let selected = handle.clone();
        handle
            .spawn(async move {
                let endpoint = Endpoint::from_shared(config.endpoint)
                    .map_err(|e| Error::invalid(e.to_string()))?
                    .connect_timeout(config.connect_timeout)
                    .buffer_size(config.limits.max_requests)
                    .initial_stream_window_size(config.http2_window_bytes)
                    .initial_connection_window_size(config.http2_window_bytes)
                    .http2_adaptive_window(false);
                let channel = endpoint
                    .connect()
                    .await
                    .map_err(|e| Error::new(ErrorKind::Transport, e.to_string()))?;
                Self::with_driver(
                    config.limits,
                    Arc::new(Server {
                        handle: selected,
                        channel: Mutex::new(Some(channel)),
                        max_message_bytes: config.max_message_bytes,
                    }),
                )
            })
            .await
            .map_err(|e| Error::new(ErrorKind::RuntimeUnavailable, e.to_string()))?
    }
}
impl Driver for Server {
    fn launch(&self, options: RequestOptions, session: WorkerSession) -> Result<()> {
        let channel = self
            .channel
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| Error::new(ErrorKind::ShuttingDown, "server client closed"))?;
        let max = self.max_message_bytes;
        let (reader, mut writer, mut guard) = session.split();
        guard.runtime_owned();
        self.handle.spawn(async move {
            let result = tokio::select! {biased;
                error=guard.cancelled()=>Err(error),
                result=run(channel,max,options,reader,&mut writer)=>result,
            };
            guard.finish(result);
        });
        Ok(())
    }
    fn shutdown(&self, completion: DriverShutdown) {
        drop(self.channel.lock().unwrap().take());
        completion.finish(Ok(()));
    }
}
struct InputStream {
    header: Option<controller::AudioStreamHeader>,
    reader: Reader,
    subscription: Subscription,
    format: AudioFormat,
    ended: bool,
    lease: Option<Lease>,
}
impl Stream for InputStream {
    type Item = controller::AudioStream;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        use controller::audio_stream::StreamPart as Part;
        self.lease.take();
        if self.ended {
            return Poll::Ready(None);
        }
        if let Some(header) = self.header.take() {
            return Poll::Ready(Some(controller::AudioStream {
                stream_part: Some(Part::AudioStreamHeader(header)),
            }));
        }
        self.subscription.register(cx.waker());
        let part = match self.reader.poll_recv(cx) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Err(_)) => {
                self.ended = true;
                return Poll::Ready(None);
            }
            Poll::Ready(Ok(None)) => {
                self.ended = true;
                Part::EndOfAudio(controller::audio_stream::EndOfAudio {})
            }
            Poll::Ready(Ok(Some(packet))) => {
                let (input, lease) = packet.into_parts();
                self.lease = Some(lease);
                match convert::encode_input(input, self.format) {
                    Ok(audio) => Part::AudioWithEmotion(audio),
                    Err(_) => {
                        self.ended = true;
                        return Poll::Ready(None);
                    }
                }
            }
        };
        Poll::Ready(Some(controller::AudioStream {
            stream_part: Some(part),
        }))
    }
}
fn transport(error: tonic::Status) -> Error {
    Error::new(
        match error.code() {
            tonic::Code::Cancelled => ErrorKind::Cancelled,
            tonic::Code::DeadlineExceeded => ErrorKind::DeadlineExceeded,
            tonic::Code::ResourceExhausted => ErrorKind::LimitExceeded,
            _ => ErrorKind::Transport,
        },
        error.to_string(),
    )
}
fn protocol(message: &str) -> Error {
    Error::new(ErrorKind::Protocol, message)
}
async fn run(
    channel: Channel,
    max: usize,
    options: RequestOptions,
    reader: Reader,
    writer: &mut Writer,
) -> Result<()> {
    let format = options.input_format;
    let encoded = convert::encode_request(options)?;
    let subscription = reader.subscription();
    let mut request = tonic::Request::new(InputStream {
        header: Some(encoded.header),
        reader,
        subscription,
        format,
        ended: false,
        lease: None,
    });
    if let Some(timeout) = encoded.timeout {
        request.set_timeout(timeout);
    }
    let mut client = A2fControllerServiceClient::new(channel)
        .max_decoding_message_size(max)
        .max_encoding_message_size(max);
    let mut response = client
        .process_audio_stream(request)
        .await
        .map_err(transport)?
        .into_inner();
    let mut info = None;
    let mut final_success = false;
    let mut last_curve = None;
    let mut audio_end = SamplePosition(0);
    while let Some(message) = response.message().await.map_err(transport)? {
        use controller::animation_data_stream::StreamPart as Part;
        let part = message
            .stream_part
            .ok_or_else(|| protocol("missing response part"))?;
        if info.is_none()
            && !matches!(&part, Part::AnimationDataStreamHeader(_))
            && !matches!(&part, Part::Status(status) if status.code == 3)
        {
            return Err(protocol("response before header"));
        }
        final_success = false;
        match part {
            Part::AnimationDataStreamHeader(header) => {
                if info.is_some() {
                    return Err(protocol("duplicate response header"));
                }
                let decoded = convert::decode_stream_info(header, LayoutId(0))?;
                writer
                    .emit(OutputEvent::StreamInfo(decoded.clone()))
                    .await?;
                info = Some(decoded);
            }
            Part::AnimationData(data) => {
                let batch = convert::decode_animation(
                    data,
                    info.as_ref()
                        .ok_or_else(|| protocol("data before header"))?,
                )?;
                if let Some(audio) = batch.audio {
                    if audio.position() != audio_end {
                        return Err(protocol("discontinuous response audio"));
                    }
                    audio_end = audio
                        .position()
                        .checked_add(audio.pcm().sample_frames(audio.format())?)?;
                    writer.emit(OutputEvent::Audio(audio)).await?;
                }
                for curve in batch.curves {
                    if last_curve.is_some_and(|t| curve.time() <= t) {
                        return Err(protocol("non-increasing response curve time"));
                    }
                    last_curve = Some(curve.time());
                    writer.emit(OutputEvent::Curves(curve)).await?;
                }
                if let Some(emotion) = batch.emotion {
                    writer.emit(OutputEvent::Emotion(emotion)).await?;
                }
                for diagnostic in batch.diagnostics {
                    writer.emit(OutputEvent::Diagnostic(diagnostic)).await?;
                }
            }
            Part::Event(event) => {
                if info.is_none() {
                    return Err(protocol("event before header"));
                }
                if event.event_type != 0 {
                    return Err(Error::new(ErrorKind::Unsupported, "unknown response event"));
                }
                writer.emit(OutputEvent::ProcessingFinished).await?;
            }
            Part::Status(status) => match status.code {
                0 => {
                    final_success = true;
                }
                1 | 2 => {
                    writer
                        .emit(OutputEvent::Diagnostic(Diagnostic {
                            severity: if status.code == 1 {
                                Severity::Info
                            } else {
                                Severity::Warning
                            },
                            message: status.message,
                        }))
                        .await?;
                }
                3 => return Err(Error::new(ErrorKind::Inference, status.message)),
                _ => return Err(protocol("unknown response status")),
            },
        }
    }
    // message(None) includes validation of the final gRPC trailers.
    if info.is_none() || !final_success {
        return Err(Error::new(
            ErrorKind::IncompleteResponse,
            "response ended without header and final SUCCESS",
        ));
    }
    Ok(())
}
