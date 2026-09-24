/// Stable error categories shared by the library and CLI.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("configuration: {0}")]
    Config(String),
    #[error("input: {0}")]
    Input(String),
    #[error("I/O or output limit: {0}")]
    Output(String),
}
pub type Result<T> = std::result::Result<T, Error>;
impl Error {
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Config(_) => 2,
            Self::Input(_) => 3,
            Self::Output(_) => 4,
        }
    }
}
