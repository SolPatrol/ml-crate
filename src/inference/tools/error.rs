//! Tool error types
//!
//! Defines error types for the tool system.

use std::fmt;
use thiserror::Error;

/// Errors that can occur during tool operations
#[derive(Debug, Error)]
pub enum ToolError {
    /// Tool not found in the registry
    #[error("Tool not found: {0}")]
    NotFound(String),

    /// Invalid arguments provided to a tool
    #[error("Invalid arguments: {0}")]
    InvalidArgs(String),

    /// Tool execution failed
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    /// JSON serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

impl ToolError {
    /// Create a not found error
    pub fn not_found(name: impl Into<String>) -> Self {
        Self::NotFound(name.into())
    }

    /// Create an invalid args error
    pub fn invalid_args(msg: impl Into<String>) -> Self {
        Self::InvalidArgs(msg.into())
    }

    /// Create an execution failed error
    pub fn execution_failed(msg: impl Into<String>) -> Self {
        Self::ExecutionFailed(msg.into())
    }

    /// Create a serialization error
    pub fn serialization(msg: impl Into<String>) -> Self {
        Self::SerializationError(msg.into())
    }
}

/// Error category for tool errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolErrorKind {
    /// Tool not found
    NotFound,
    /// Invalid arguments
    InvalidArgs,
    /// Execution failed
    ExecutionFailed,
    /// Serialization error
    Serialization,
}

impl ToolError {
    /// Get the kind of error
    pub fn kind(&self) -> ToolErrorKind {
        match self {
            Self::NotFound(_) => ToolErrorKind::NotFound,
            Self::InvalidArgs(_) => ToolErrorKind::InvalidArgs,
            Self::ExecutionFailed(_) => ToolErrorKind::ExecutionFailed,
            Self::SerializationError(_) => ToolErrorKind::Serialization,
        }
    }

    /// Check if the error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(self, Self::ExecutionFailed(_))
    }
}

impl fmt::Display for ToolErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolErrorKind::NotFound => write!(f, "NotFound"),
            ToolErrorKind::InvalidArgs => write!(f, "InvalidArgs"),
            ToolErrorKind::ExecutionFailed => write!(f, "ExecutionFailed"),
            ToolErrorKind::Serialization => write!(f, "Serialization"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not_found_error_display() {
        let err = ToolError::not_found("my_tool");
        assert_eq!(err.to_string(), "Tool not found: my_tool");
    }

    #[test]
    fn test_invalid_args_error_display() {
        let err = ToolError::invalid_args("missing required field 'name'");
        assert_eq!(
            err.to_string(),
            "Invalid arguments: missing required field 'name'"
        );
    }

    #[test]
    fn test_execution_failed_error_display() {
        let err = ToolError::execution_failed("database connection failed");
        assert_eq!(
            err.to_string(),
            "Execution failed: database connection failed"
        );
    }

    #[test]
    fn test_serialization_error_display() {
        let err = ToolError::serialization("invalid JSON");
        assert_eq!(err.to_string(), "Serialization error: invalid JSON");
    }

    #[test]
    fn test_error_kind() {
        assert_eq!(
            ToolError::not_found("x").kind(),
            ToolErrorKind::NotFound
        );
        assert_eq!(
            ToolError::invalid_args("x").kind(),
            ToolErrorKind::InvalidArgs
        );
        assert_eq!(
            ToolError::execution_failed("x").kind(),
            ToolErrorKind::ExecutionFailed
        );
        assert_eq!(
            ToolError::serialization("x").kind(),
            ToolErrorKind::Serialization
        );
    }

    #[test]
    fn test_is_recoverable() {
        assert!(!ToolError::not_found("x").is_recoverable());
        assert!(!ToolError::invalid_args("x").is_recoverable());
        assert!(ToolError::execution_failed("x").is_recoverable());
        assert!(!ToolError::serialization("x").is_recoverable());
    }

    #[test]
    fn test_error_kind_display() {
        assert_eq!(ToolErrorKind::NotFound.to_string(), "NotFound");
        assert_eq!(ToolErrorKind::InvalidArgs.to_string(), "InvalidArgs");
        assert_eq!(
            ToolErrorKind::ExecutionFailed.to_string(),
            "ExecutionFailed"
        );
        assert_eq!(ToolErrorKind::Serialization.to_string(), "Serialization");
    }
}
