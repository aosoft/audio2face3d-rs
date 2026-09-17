//! Application supplied, statically dispatched authentication.
mod bearer;
pub(crate) mod gate;
pub use audio2face3d::types::RequestId;
pub use bearer::{InvalidApiKey, validate_api_key};
use std::{
    fmt,
    future::{Future, Ready, ready},
    net::SocketAddr,
};

/// A borrowed credential. Formatting never exposes its contents.
#[derive(Clone, Copy)]
pub struct SecretApiKey<'a>(&'a str);
impl<'a> SecretApiKey<'a> {
    /// Intentionally expose the credential to the application's verifier.
    pub fn expose(self) -> &'a str {
        self.0
    }
}
impl fmt::Debug for SecretApiKey<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[REDACTED]")
    }
}
impl fmt::Display for SecretApiKey<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RpcMethod {
    ProcessAudioStream,
    HealthCheck,
    HealthWatch,
}
#[derive(Clone, Copy, Debug)]
pub struct AuthRequest<'a> {
    pub api_key: SecretApiKey<'a>,
    pub request_id: RequestId,
    pub method: RpcMethod,
    pub peer_addr: Option<SocketAddr>,
}
/// A safe application identity, never a credential or its prefix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Principal {
    subject: String,
}
impl Principal {
    pub fn new(subject: impl Into<String>) -> Result<Self, AuthError> {
        let subject = subject.into();
        if subject.is_empty() || subject.len() > 128 || subject.chars().any(char::is_control) {
            return Err(AuthError::Unavailable);
        }
        Ok(Self { subject })
    }
    pub fn subject(&self) -> &str {
        &self.subject
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthError {
    InvalidCredential,
    Forbidden,
    Unavailable,
}
impl fmt::Display for AuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for AuthError {}
pub type AuthResult = Result<Principal, AuthError>;
pub trait Authenticator: Send + Sync + 'static {
    type Future<'a>: Future<Output = AuthResult> + Send + 'a
    where
        Self: 'a;
    fn authenticate<'a>(&'a self, request: AuthRequest<'a>) -> Self::Future<'a>;
}
impl<F> Authenticator for F
where
    F: for<'a> Fn(AuthRequest<'a>) -> AuthResult + Send + Sync + 'static,
{
    type Future<'a>
        = Ready<AuthResult>
    where
        Self: 'a;
    fn authenticate<'a>(&'a self, request: AuthRequest<'a>) -> Self::Future<'a> {
        ready(self(request))
    }
}
/// No authenticator is installed; this marker cannot grant access by invocation.
pub struct NoAuth {
    _private: (),
}
impl Authenticator for NoAuth {
    type Future<'a> = Ready<AuthResult>;
    fn authenticate<'a>(&'a self, _: AuthRequest<'a>) -> Self::Future<'a> {
        ready(Err(AuthError::Unavailable))
    }
}
pub struct AsyncAuthenticator<F>(F);
pub fn async_authenticator<F, Fut>(f: F) -> AsyncAuthenticator<F>
where
    F: for<'a> Fn(AuthRequest<'a>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = AuthResult> + Send + 'static,
{
    AsyncAuthenticator(f)
}
impl<F, Fut> Authenticator for AsyncAuthenticator<F>
where
    F: for<'a> Fn(AuthRequest<'a>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = AuthResult> + Send + 'static,
{
    type Future<'a>
        = Fut
    where
        Self: 'a;
    fn authenticate<'a>(&'a self, request: AuthRequest<'a>) -> Self::Future<'a> {
        (self.0)(request)
    }
}
