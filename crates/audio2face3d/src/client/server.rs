use crate::client::{Client, Limits, core::Lease, driver::*, notify::Subscription, types::*};
use crate::protocol::{
    convert,
    wire::{
        controller,
        nvidia_ace::services::a2f_controller::v1::a2f_controller_service_client::A2fControllerServiceClient,
    },
};
use crate::{Audio2Face3DContext, logging::integration::LogScope};
use std::{
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
    time::Duration,
};
use tokio::runtime::Handle;
use tokio_stream::Stream;
use tonic::transport::{Channel, Endpoint};

#[derive(Clone)]
pub struct ServerConfig {
    pub(crate) endpoint: String,
    /// Optional Bearer credential sent on every inference RPC. Debug redacts the value.
    /// None sends no authorization header; empty or malformed values are rejected.
    pub(crate) api_key: Option<String>,
    /// None selects try_current during initialization; no runtime is created.
    pub(crate) runtime: Option<Handle>,
    pub(crate) limits: Limits,
    pub(crate) connect_timeout: Duration,
    pub(crate) max_message_bytes: usize,
    /// Fixed HTTP/2 connection and stream receive windows; adaptive growth is disabled.
    pub(crate) http2_window_bytes: u32,
}
impl ServerConfig {
    pub fn validate(&self) -> Result<()> {
        authorization(self.api_key.as_deref())?;
        self.limits.validate()?;
        if self.connect_timeout.is_zero()
            || self.max_message_bytes == 0
            || !(65_535..=0x7fff_ffff).contains(&self.http2_window_bytes)
        {
            return Err(Error::invalid("invalid server transport limits"));
        }
        let uri: tonic::codegen::http::Uri = self
            .endpoint
            .parse()
            .map_err(|_| Error::invalid("invalid server endpoint"))?;
        if !matches!(uri.scheme_str(), Some("http" | "https")) || uri.host().is_none() {
            return Err(Error::invalid("invalid server endpoint"));
        }
        Ok(())
    }

    pub(crate) fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            api_key: None,
            runtime: None,
            limits: Limits::default(),
            connect_timeout: Duration::from_secs(10),
            max_message_bytes: 4 * 1024 * 1024,
            http2_window_bytes: 65_535,
        }
    }
}
impl std::fmt::Debug for ServerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerConfig")
            .field("endpoint", &self.endpoint)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("runtime", &self.runtime)
            .field("limits", &self.limits)
            .field("connect_timeout", &self.connect_timeout)
            .field("max_message_bytes", &self.max_message_bytes)
            .field("http2_window_bytes", &self.http2_window_bytes)
            .finish()
    }
}
type Authorization = tonic::metadata::MetadataValue<tonic::metadata::Ascii>;
fn authorization(key: Option<&str>) -> Result<Option<Authorization>> {
    let Some(key) = key else { return Ok(None) };
    let invalid = || Error::invalid("invalid API key format");
    // RFC 6750 b64token; preserve the credential exactly, without trimming.
    let body = key.trim_end_matches('=');
    if key.is_empty()
        || key.len() > 4096
        || body.is_empty()
        || !body
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"-._~+/".contains(&c))
    {
        return Err(invalid());
    }
    let mut value: Authorization = format!("Bearer {key}").parse().map_err(|_| invalid())?;
    value.set_sensitive(true);
    Ok(Some(value))
}
struct Server {
    handle: Handle,
    channel: Mutex<Option<Channel>>,
    authorization: Option<Authorization>,
    max_message_bytes: usize,
}
impl Client {
    async fn server_inner(mut config: ServerConfig) -> Result<Self> {
        let mut observation =
            crate::logging::operation::Operation::new("remote connection finished", module_path!());
        let result = async {
            crate::logging::integration::log(crate::logging::LogLevel::Info, || {
                crate::logging::LogRecord::new("connecting remote inference client")
                    .field("source", module_path!())
            });
            config.validate()?;
            let authorization = authorization(config.api_key.take().as_deref())?;
            let handle = config
                .runtime
                .or_else(|| Handle::try_current().ok())
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::RuntimeUnavailable,
                        "server mode requires a driven Tokio runtime with I/O and time enabled",
                    )
                })?;
            observation.stage = "connect";
            let selected = handle.clone();
            handle
                .spawn(LogScope::capture().wrap_future(async move {
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
                            authorization,
                            max_message_bytes: config.max_message_bytes,
                        }),
                    )
                }))
                .await
                .map_err(|e| Error::new(ErrorKind::RuntimeUnavailable, e.to_string()))?
        }
        .await;
        observation.finish(&result);
        result
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
        let authorization = self.authorization.clone();
        let (reader, mut writer, mut guard) = session.split();
        guard.runtime_owned();
        self.handle
            .spawn(LogScope::capture().wrap_future(async move {
                let mut observation = crate::logging::operation::Operation::new(
                    "remote inference request finished",
                    module_path!(),
                );
                observation.stage = "streaming";
                let result = tokio::select! {biased;
                    error=guard.cancelled()=>Err(error),
                    result=run(channel,max,authorization,options,reader,&mut writer)=>result,
                };
                observation.finish(&result);
                guard.finish(result);
            }));
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
    authorization: Option<Authorization>,
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
    if let Some(value) = authorization {
        request.metadata_mut().insert("authorization", value);
    }
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

impl Client {
    pub async fn server(config: ServerConfig) -> Result<Self> {
        let scope = LogScope::capture();
        scope.wrap_future(Self::server_inner(config)).await
    }
    pub async fn server_with_context(
        config: ServerConfig,
        context: Audio2Face3DContext,
    ) -> Result<Self> {
        let scope = LogScope::new(context);
        scope.wrap_future(Self::server_inner(config)).await
    }
}

#[cfg(test)]
mod auth_tests {
    use super::*;
    #[test]
    fn bearer_value_preserves_tokens_and_is_sensitive() {
        assert!(authorization(None).unwrap().is_none());
        for key in ["a-._~+/", "abc==", &"a".repeat(4096)] {
            let value = authorization(Some(key)).unwrap().unwrap();
            assert_eq!(value.to_str().unwrap(), format!("Bearer {key}"));
            assert!(value.is_sensitive());
        }
        for key in [" a", "a ", "a,b", "a\tb", "=abc", "ab=c"] {
            assert_eq!(
                authorization(Some(key)).unwrap_err().kind(),
                ErrorKind::InvalidInput
            );
        }
    }
}

/// Consuming builder; validation runs in build before resources are started.
#[derive(Clone, Debug)]
#[must_use]
pub struct ServerConfigBuilder {
    config: ServerConfig,
}
impl ServerConfig {
    pub fn builder(endpoint: impl Into<String>) -> ServerConfigBuilder {
        ServerConfigBuilder {
            config: ServerConfig::new(endpoint),
        }
    }
}
impl ServerConfigBuilder {
    pub fn endpoint(mut self, value: impl Into<String>) -> Self {
        self.config.endpoint = value.into();
        self
    }
    pub fn api_key(mut self, value: impl Into<String>) -> Self {
        self.config.api_key = Some(value.into());
        self
    }
    pub fn optional_api_key(mut self, value: Option<String>) -> Self {
        self.config.api_key = value;
        self
    }
    pub fn runtime(mut self, value: Handle) -> Self {
        self.config.runtime = Some(value);
        self
    }
    pub fn optional_runtime(mut self, value: Option<Handle>) -> Self {
        self.config.runtime = value;
        self
    }
    pub fn limits(mut self, value: Limits) -> Self {
        self.config.limits = value;
        self
    }
    pub fn connect_timeout(mut self, value: Duration) -> Self {
        self.config.connect_timeout = value;
        self
    }
    pub fn max_message_bytes(mut self, value: usize) -> Self {
        self.config.max_message_bytes = value;
        self
    }
    pub fn http2_window_bytes(mut self, value: u32) -> Self {
        self.config.http2_window_bytes = value;
        self
    }
    pub fn build(self) -> Result<ServerConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl ServerConfig {
    pub fn into_builder(self) -> ServerConfigBuilder {
        ServerConfigBuilder { config: self }
    }
    pub fn endpoint(&self) -> &String {
        &self.endpoint
    }
    pub fn api_key(&self) -> &Option<String> {
        &self.api_key
    }
    pub fn runtime(&self) -> &Option<Handle> {
        &self.runtime
    }
    pub fn limits(&self) -> &Limits {
        &self.limits
    }
    pub fn connect_timeout(&self) -> Duration {
        self.connect_timeout
    }
    pub fn max_message_bytes(&self) -> usize {
        self.max_message_bytes
    }
    pub fn http2_window_bytes(&self) -> u32 {
        self.http2_window_bytes
    }
}
