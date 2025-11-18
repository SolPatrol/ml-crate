//! Error types for the Candle Adapter

use thiserror::Error;

/// Errors that can occur during Candle adapter operations
#[derive(Debug, Error)]
pub enum CandleAdapterError {
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

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Model not loaded
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Generic error from anyhow
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

// Note: anyhow::Error automatically implements From<T> for any T: StdError + Send + Sync + 'static
// So we don't need a manual From implementation - thiserror handles this for us

/// Result type for Candle adapter operations
pub type Result<T> = std::result::Result<T, CandleAdapterError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CandleAdapterError::InferenceFailed("model error".to_string());
        assert_eq!(err.to_string(), "Inference failed: model error");

        let err = CandleAdapterError::ContextTooLong {
            actual: 10000,
            max: 8192,
        };
        assert_eq!(
            err.to_string(),
            "Context too long: 10000 tokens > 8192 max"
        );
    }

    #[test]
    fn test_error_conversion() {
        let err = CandleAdapterError::RateLimitExceeded;
        let anyhow_err: anyhow::Error = err.into();
        assert!(anyhow_err.to_string().contains("Rate limit exceeded"));
    }
}
