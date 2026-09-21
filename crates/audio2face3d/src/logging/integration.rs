//! Scope propagation support for cooperating libraries. Applications normally
//! only need Audio2Face3DContext and Logger.
pub use super::scope::{LogScope, ScopeGuard, Scoped};

/// Checks the current application logger before preparing diagnostic data.
pub fn enabled(level: super::LogLevel) -> bool {
    level != super::LogLevel::Off && level >= LogScope::capture().context().logger().log_level()
}
