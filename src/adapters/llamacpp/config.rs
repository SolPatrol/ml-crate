//! Configuration for the LlamaCpp Adapter

use serde::{Deserialize, Serialize};

/// Configuration for LlamaCppAdapter
///
/// Controls inference parameters and behavior for the llama.cpp-based adapter.
/// Mirrors CandleConfig structure with llama.cpp-specific additions (repeat_penalty).
///
/// # Example
///
/// ```rust
/// use ml_crate_dsrs::adapters::llamacpp::LlamaCppConfig;
///
/// // Use default configuration
/// let config = LlamaCppConfig::default();
///
/// // Or customize with builder pattern
/// let config = LlamaCppConfig::new("qwen2.5-0.5b")
///     .with_max_tokens(1024)
///     .with_temperature(0.8)
///     .with_repeat_penalty(1.2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppConfig {
    /// Model identifier (used for logging and debugging)
    pub model_name: String,

    /// Maximum tokens to generate
    pub max_tokens: usize,

    /// Sampling temperature (0.0 - 2.0)
    /// Lower values make output more deterministic
    pub temperature: f32,

    /// Top-p nucleus sampling
    /// Probability mass to consider for sampling
    pub top_p: f32,

    /// Top-k sampling
    /// If Some(k), only consider top k tokens
    pub top_k: Option<usize>,

    /// Repeat penalty (llama.cpp specific, 1.0 = disabled)
    /// Higher values penalize token repetition more strongly
    pub repeat_penalty: f32,

    /// Context window size
    /// Maximum number of tokens the model can process
    pub context_length: usize,

    // Production features (matching CandleConfig)
    /// Maximum total tokens allowed across all requests
    pub token_budget_limit: Option<usize>,

    /// Rate limiting: requests per minute
    pub requests_per_minute: Option<u32>,

    /// Number of retry attempts on transient failures
    pub max_retries: u32,

    /// Initial backoff delay in milliseconds
    pub initial_backoff_ms: u64,

    /// Maximum backoff delay in milliseconds
    pub max_backoff_ms: u64,

    /// Enable response caching
    pub enable_cache: bool,

    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,

    /// Enable token-by-token streaming output
    pub enable_streaming: bool,

    /// Random seed for sampling (None = random)
    pub seed: Option<u64>,
}

impl Default for LlamaCppConfig {
    /// Default configuration for Qwen2.5-0.5B model
    fn default() -> Self {
        Self {
            model_name: "llama-qwen2.5-0.5b".to_string(),
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: None,
            repeat_penalty: 1.1, // llama.cpp specific
            context_length: 32768, // Qwen2.5-0.5B supports 32K context
            token_budget_limit: None,
            requests_per_minute: None,
            max_retries: 3,
            initial_backoff_ms: 100,
            max_backoff_ms: 5000,
            enable_cache: false,
            cache_ttl_secs: 300,
            enable_streaming: false,
            seed: None,
        }
    }
}

impl LlamaCppConfig {
    /// Create a new configuration with custom model name
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            ..Default::default()
        }
    }

    /// Set maximum tokens to generate
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set sampling temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-p sampling
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set top-k sampling
    pub fn with_top_k(mut self, top_k: Option<usize>) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set repeat penalty (llama.cpp specific)
    pub fn with_repeat_penalty(mut self, repeat_penalty: f32) -> Self {
        self.repeat_penalty = repeat_penalty;
        self
    }

    /// Set context window size
    pub fn with_context_length(mut self, context_length: usize) -> Self {
        self.context_length = context_length;
        self
    }

    /// Enable retry logic with custom attempts
    pub fn with_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Enable response caching
    pub fn with_cache(mut self, enable: bool, ttl_secs: u64) -> Self {
        self.enable_cache = enable;
        self.cache_ttl_secs = ttl_secs;
        self
    }

    /// Enable streaming output
    pub fn with_streaming(mut self, enable: bool) -> Self {
        self.enable_streaming = enable;
        self
    }

    /// Set token budget limit
    pub fn with_token_budget(mut self, limit: Option<usize>) -> Self {
        self.token_budget_limit = limit;
        self
    }

    /// Set rate limiting
    pub fn with_rate_limit(mut self, requests_per_minute: Option<u32>) -> Self {
        self.requests_per_minute = requests_per_minute;
        self
    }

    /// Set random seed for reproducible sampling
    pub fn with_seed(mut self, seed: Option<u64>) -> Self {
        self.seed = seed;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LlamaCppConfig::default();
        assert_eq!(config.model_name, "llama-qwen2.5-0.5b");
        assert_eq!(config.max_tokens, 512);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.repeat_penalty, 1.1);
        assert_eq!(config.context_length, 32768);
    }

    #[test]
    fn test_config_builder() {
        let config = LlamaCppConfig::new("custom-model")
            .with_max_tokens(1024)
            .with_temperature(0.8)
            .with_top_k(Some(50))
            .with_repeat_penalty(1.2)
            .with_context_length(8192)
            .with_cache(true, 600);

        assert_eq!(config.model_name, "custom-model");
        assert_eq!(config.max_tokens, 1024);
        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.top_k, Some(50));
        assert_eq!(config.repeat_penalty, 1.2);
        assert_eq!(config.context_length, 8192);
        assert!(config.enable_cache);
        assert_eq!(config.cache_ttl_secs, 600);
    }
}
