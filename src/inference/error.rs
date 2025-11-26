//! DSPy Engine error types
//!
//! Defines all error types used by the DSPy Engine inference system.

use std::fmt;
use thiserror::Error;

use super::tools::ToolError;

/// Result type for DSPy Engine operations
pub type Result<T> = std::result::Result<T, DSPyEngineError>;

/// Errors that can occur during DSPy Engine operations
#[derive(Error, Debug)]
pub enum DSPyEngineError {
    /// Module not found in the registry
    #[error("Module not found: {0}")]
    ModuleNotFound(String),

    /// Signature not found in the registry
    #[error("Signature not found: {0}")]
    SignatureNotFound(String),

    /// Tools required but not enabled for this module
    #[error("Tools not enabled for module: {0}")]
    ToolsNotEnabled(String),

    /// Maximum iterations reached during tool execution
    #[error("Maximum iterations ({0}) reached during tool execution")]
    MaxIterationsReached(usize),

    /// Inference error during model execution
    #[error("Inference error: {0}")]
    InferenceError(String),

    /// Parse error when reading module files
    #[error("Parse error: {0}")]
    ParseError(String),

    /// I/O error when reading files
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Runtime error during module execution
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Module file hash mismatch
    #[error("Module hash mismatch for '{module}': expected {expected}, got {actual}")]
    HashMismatch {
        module: String,
        expected: String,
        actual: String,
    },

    /// Invalid module format
    #[error("Invalid module format: {0}")]
    InvalidModuleFormat(String),

    /// File watcher error (hot reload)
    #[error("File watcher error: {0}")]
    WatcherError(String),

    /// Hot reload failed
    #[error("Hot reload failed for {path}: {reason}")]
    HotReloadFailed { path: String, reason: String },

    /// Tool execution error
    #[error("Tool error: {0}")]
    ToolError(#[from] ToolError),
}

impl DSPyEngineError {
    /// Create a module not found error
    pub fn module_not_found(name: impl Into<String>) -> Self {
        Self::ModuleNotFound(name.into())
    }

    /// Create a signature not found error
    pub fn signature_not_found(name: impl Into<String>) -> Self {
        Self::SignatureNotFound(name.into())
    }

    /// Create an inference error
    pub fn inference(msg: impl Into<String>) -> Self {
        Self::InferenceError(msg.into())
    }

    /// Create a parse error
    pub fn parse(msg: impl Into<String>) -> Self {
        Self::ParseError(msg.into())
    }

    /// Create a runtime error
    pub fn runtime(msg: impl Into<String>) -> Self {
        Self::RuntimeError(msg.into())
    }

    /// Create a config error
    pub fn config(msg: impl Into<String>) -> Self {
        Self::ConfigError(msg.into())
    }

    /// Create a watcher error
    pub fn watcher(msg: impl Into<String>) -> Self {
        Self::WatcherError(msg.into())
    }

    /// Create a hot reload failed error
    pub fn hot_reload_failed(path: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::HotReloadFailed {
            path: path.into(),
            reason: reason.into(),
        }
    }
}

/// ErrorKind for categorizing errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Module-related errors (not found, invalid format)
    Module,
    /// Signature-related errors
    Signature,
    /// Inference/execution errors
    Inference,
    /// File I/O errors
    Io,
    /// Configuration errors
    Config,
    /// Hot reload errors
    HotReload,
}

impl DSPyEngineError {
    /// Get the kind of error
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::ModuleNotFound(_) | Self::InvalidModuleFormat(_) | Self::HashMismatch { .. } => {
                ErrorKind::Module
            }
            Self::SignatureNotFound(_) => ErrorKind::Signature,
            Self::InferenceError(_)
            | Self::RuntimeError(_)
            | Self::ToolsNotEnabled(_)
            | Self::MaxIterationsReached(_)
            | Self::ToolError(_) => ErrorKind::Inference,
            Self::IoError(_) | Self::JsonError(_) | Self::ParseError(_) => ErrorKind::Io,
            Self::ConfigError(_) => ErrorKind::Config,
            Self::WatcherError(_) | Self::HotReloadFailed { .. } => ErrorKind::HotReload,
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::MaxIterationsReached(_)
                | Self::InferenceError(_)
                | Self::RuntimeError(_)
                | Self::ToolError(_)
        )
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::Module => write!(f, "Module"),
            ErrorKind::Signature => write!(f, "Signature"),
            ErrorKind::Inference => write!(f, "Inference"),
            ErrorKind::Io => write!(f, "I/O"),
            ErrorKind::Config => write!(f, "Config"),
            ErrorKind::HotReload => write!(f, "HotReload"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = DSPyEngineError::module_not_found("test_module");
        assert_eq!(err.to_string(), "Module not found: test_module");

        let err = DSPyEngineError::signature_not_found("TestSignature");
        assert_eq!(err.to_string(), "Signature not found: TestSignature");

        let err = DSPyEngineError::MaxIterationsReached(10);
        assert_eq!(
            err.to_string(),
            "Maximum iterations (10) reached during tool execution"
        );
    }

    #[test]
    fn test_error_kind() {
        assert_eq!(
            DSPyEngineError::module_not_found("x").kind(),
            ErrorKind::Module
        );
        assert_eq!(
            DSPyEngineError::signature_not_found("x").kind(),
            ErrorKind::Signature
        );
        assert_eq!(
            DSPyEngineError::inference("x").kind(),
            ErrorKind::Inference
        );
        assert_eq!(
            DSPyEngineError::IoError(std::io::Error::new(std::io::ErrorKind::NotFound, "test"))
                .kind(),
            ErrorKind::Io
        );
    }

    #[test]
    fn test_error_is_recoverable() {
        assert!(DSPyEngineError::MaxIterationsReached(5).is_recoverable());
        assert!(DSPyEngineError::inference("test").is_recoverable());
        assert!(!DSPyEngineError::module_not_found("test").is_recoverable());
        assert!(!DSPyEngineError::signature_not_found("test").is_recoverable());
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: DSPyEngineError = io_err.into();
        assert!(matches!(err, DSPyEngineError::IoError(_)));
    }

    #[test]
    fn test_from_json_error() {
        let json_result: serde_json::Result<String> = serde_json::from_str("invalid");
        if let Err(json_err) = json_result {
            let err: DSPyEngineError = json_err.into();
            assert!(matches!(err, DSPyEngineError::JsonError(_)));
        }
    }

    #[test]
    fn test_hash_mismatch_display() {
        let err = DSPyEngineError::HashMismatch {
            module: "test_module".to_string(),
            expected: "abc123".to_string(),
            actual: "def456".to_string(),
        };
        assert!(err.to_string().contains("test_module"));
        assert!(err.to_string().contains("abc123"));
        assert!(err.to_string().contains("def456"));
    }
}
