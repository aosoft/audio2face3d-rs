use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RequestId(pub u64);

/// Counters distinguish transport receipt from delivery to the application.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Progress {
    pub received_audio_bytes: u64,
    pub delivered_audio_bytes: u64,
    pub received_curve_frames: u64,
    pub delivered_curve_frames: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorKind {
    InvalidInput,
    Unsupported,
    QueueFull,
    LimitExceeded,
    DeadlineExceeded,
    Cancelled,
    Transport,
    Protocol,
    IncompleteResponse,
    Inference,
    ShuttingDown,
    RuntimeUnavailable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    kind: ErrorKind,
    message: String,
    request_id: Option<RequestId>,
    progress: Progress,
}
impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            request_id: None,
            progress: Progress::default(),
        }
    }
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidInput, message)
    }
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
    pub fn message(&self) -> &str {
        &self.message
    }
    pub fn request_id(&self) -> Option<RequestId> {
        self.request_id
    }
    pub fn progress(&self) -> Progress {
        self.progress
    }
    pub fn with_request(mut self, id: RequestId) -> Self {
        self.request_id = Some(id);
        self
    }
    pub fn with_progress(mut self, progress: Progress) -> Self {
        self.progress = progress;
        self
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}
impl std::error::Error for Error {}
