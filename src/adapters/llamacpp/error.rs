//! Error types for the LlamaCpp Adapter

use thiserror::Error;

/// Errors that can occur during LlamaCpp adapter operations
#[derive(Debug, Error)]
pub enum LlamaCppError {
    /// Inference operation failed
    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    /// Tokenization failed
    #[error("Tokenization failed: {0}")]
    TokenizationFailed(String),

    /// Context exceeds maximum length
    #[error("Context too long: {actual} tokens > {max} max")]
    ContextTooLong {
        /// Actual number of tokens
        actual: usize,
        /// Maximum allowed tokens
        max: usize,
    },

    /// Token budget exhausted
    #[error("Token budget exhausted: {used}/{limit}")]
    TokenBudgetExhausted {
        /// Tokens used so far
        used: usize,
        /// Token budget limit
        limit: usize,
    },

    /// Rate limit exceeded
    #[error("Rate limit exceeded: too many requests")]
    RateLimitExceeded,

    /// Model not loaded
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    /// Backend error (llama.cpp specific)
    #[error("Backend error: {0}")]
    BackendError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Generic error from anyhow
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Result type for LlamaCpp adapter operations
pub type Result<T> = std::result::Result<T, LlamaCppError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = LlamaCppError::InferenceFailed("model error".to_string());
        assert_eq!(err.to_string(), "Inference failed: model error");

        let err = LlamaCppError::ContextTooLong {
            actual: 10000,
            max: 8192,
        };
        assert_eq!(
            err.to_string(),
            "Context too long: 10000 tokens > 8192 max"
        );

        let err = LlamaCppError::BackendError("Vulkan init failed".to_string());
        assert_eq!(err.to_string(), "Backend error: Vulkan init failed");
    }

    #[test]
    fn test_error_conversion() {
        let err = LlamaCppError::RateLimitExceeded;
        let anyhow_err: anyhow::Error = err.into();
        assert!(anyhow_err.to_string().contains("Rate limit exceeded"));
    }
}
