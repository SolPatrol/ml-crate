//! Model Pool - Manages loading and lifecycle of Candle models
//!
//! The Model Pool is responsible for:
//! - Loading models from disk (safetensors files)
//! - Managing tokenizers
//! - Device selection (CPU, CUDA, Metal)
//! - Model caching (avoid reloading)
//!
//! # Phase 1 Scope
//!
//! - Basic model loading from disk
//! - Device detection and selection
//! - Simple caching (HashMap)
//!
//! # Future Phases
//!
//! - VRAM management
//! - Model quantization (Q4, Q8)
//! - Dynamic model unloading
//! - HuggingFace Hub downloads

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM as Qwen2Model};
use tokenizers::Tokenizer;

use crate::adapters::candle::adapter::LoadedModel;
use crate::adapters::candle::error::{CandleAdapterError, Result};

/// ModelPool manages the lifecycle of loaded Candle models
///
/// # Example
///
/// ```rust,ignore
/// use ml_crate_dsrs::model_pool::ModelPool;
///
/// # async fn example() -> Result<()> {
/// let pool = ModelPool::new("./models".into());
///
/// // Load model (will cache for reuse)
/// let loaded = pool.load_model("Qwen3-0.6B").await?;
///
/// // Second load is instant (from cache)
/// let loaded_again = pool.load_model("Qwen3-0.6B").await?;
/// # Ok(())
/// # }
/// ```
pub struct ModelPool {
    /// Cache of loaded models (model_name -> LoadedModel)
    cache: Arc<RwLock<HashMap<String, Arc<LoadedModel>>>>,

    /// Base directory where models are stored
    models_dir: PathBuf,
}

impl ModelPool {
    /// Create a new ModelPool
    ///
    /// # Arguments
    ///
    /// * `models_dir` - Directory containing model subdirectories
    ///
    /// # Example
    ///
    /// ```rust
    /// use ml_crate_dsrs::model_pool::ModelPool;
    ///
    /// let pool = ModelPool::new("./models".into());
    /// ```
    pub fn new(models_dir: PathBuf) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            models_dir,
        }
    }

    /// Load a model by name (with caching)
    ///
    /// If the model is already loaded, returns the cached version.
    /// Otherwise, loads from disk and adds to cache.
    ///
    /// # Arguments
    ///
    /// * `model_name` - Name of the model (subdirectory in models_dir)
    ///
    /// # Returns
    ///
    /// Arc-wrapped LoadedModel ready for inference
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model directory doesn't exist
    /// - Model files are corrupted
    /// - Device initialization fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// # use ml_crate_dsrs::model_pool::ModelPool;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let pool = ModelPool::new("./models".into());
    /// let loaded = pool.load_model("Qwen3-0.6B").await?;
    ///
    /// println!("Loaded model: {}", loaded.model_name);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn load_model(&self, model_name: &str) -> Result<Arc<LoadedModel>> {
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(model_name) {
                tracing::info!("Using cached model: {}", model_name);
                return Ok(Arc::clone(cached));
            }
        }

        // Not in cache, load from disk
        tracing::info!("Loading model from disk: {}", model_name);
        let model_path = self.models_dir.join(model_name);

        let loaded = Self::load_from_disk(model_path, model_name.to_string()).await?;
        let loaded = Arc::new(loaded);

        // Add to cache
        {
            let mut cache = self.cache.write().await;
            cache.insert(model_name.to_string(), Arc::clone(&loaded));
        }

        tracing::info!("Model loaded and cached: {}", model_name);
        Ok(loaded)
    }

    /// Load model from disk (CPU-bound, uses spawn_blocking)
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to model directory
    /// * `model_name` - Name for logging
    ///
    /// # Returns
    ///
    /// LoadedModel with model, tokenizer, and device
    ///
    /// # Model Directory Structure
    ///
    /// ```text
    /// model_path/
    /// ├── config.json          # Qwen2 model configuration
    /// ├── tokenizer.json       # HuggingFace tokenizer
    /// └── model.safetensors    # Model weights (F16 or F32)
    /// ```
    async fn load_from_disk(model_path: PathBuf, model_name: String) -> Result<LoadedModel> {
        tokio::task::spawn_blocking(move || {
            // 1. Initialize CUDA device (GPU-only, no CPU fallback)
            // Requires CUDA 12.x toolkit installed
            tracing::info!("Initializing CUDA GPU device");
            let device = Device::new_cuda(0).map_err(|e| {
                CandleAdapterError::ConfigError(format!(
                    "CUDA GPU initialization failed. Ensure CUDA 12.x toolkit is installed: {}",
                    e
                ))
            })?;

            // 2. Load tokenizer
            let tokenizer_path = model_path.join("tokenizer.json");
            tracing::debug!("Loading tokenizer from: {:?}", tokenizer_path);

            let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                CandleAdapterError::TokenizationFailed(format!(
                    "Failed to load tokenizer from {:?}: {}",
                    tokenizer_path, e
                ))
            })?;

            // 3. Load config
            let config_path = model_path.join("config.json");
            tracing::debug!("Loading config from: {:?}", config_path);

            // Parse config.json as raw JSON first to handle null values (e.g., sliding_window)
            let config_json: serde_json::Value = serde_json::from_reader(
                std::fs::File::open(&config_path).map_err(|e| {
                    CandleAdapterError::ConfigError(format!(
                        "Failed to open config.json at {:?}: {}",
                        config_path, e
                    ))
                })?,
            )
            .map_err(|e| {
                CandleAdapterError::ConfigError(format!("Failed to parse config.json: {}", e))
            })?;

            // Build Qwen2Config manually, handling null values
            let config = Qwen2Config {
                vocab_size: config_json["vocab_size"].as_u64().unwrap() as usize,
                hidden_size: config_json["hidden_size"].as_u64().unwrap() as usize,
                intermediate_size: config_json["intermediate_size"].as_u64().unwrap() as usize,
                num_hidden_layers: config_json["num_hidden_layers"].as_u64().unwrap() as usize,
                num_attention_heads: config_json["num_attention_heads"].as_u64().unwrap()
                    as usize,
                num_key_value_heads: config_json["num_key_value_heads"].as_u64().unwrap()
                    as usize,
                max_position_embeddings: config_json["max_position_embeddings"]
                    .as_u64()
                    .unwrap() as usize,
                max_window_layers: config_json["max_window_layers"].as_u64().unwrap() as usize,
                // Handle null sliding_window - use max_position_embeddings as default
                sliding_window: config_json["sliding_window"]
                    .as_u64()
                    .unwrap_or(config_json["max_position_embeddings"].as_u64().unwrap())
                    as usize,
                use_sliding_window: config_json["use_sliding_window"].as_bool().unwrap_or(false),
                hidden_act: candle_nn::Activation::Silu, // Qwen2/Qwen2.5 use SiLU activation
                tie_word_embeddings: config_json["tie_word_embeddings"]
                    .as_bool()
                    .unwrap_or(false),
                rms_norm_eps: config_json["rms_norm_eps"].as_f64().unwrap(),
                rope_theta: config_json["rope_theta"].as_f64().unwrap(),
            };

            // 4. Load weights
            let weights_path = model_path.join("model.safetensors");
            tracing::debug!("Loading weights from: {:?}", weights_path);

            if !weights_path.exists() {
                return Err(CandleAdapterError::ModelNotLoaded(format!(
                    "Model weights not found at {:?}",
                    weights_path
                )));
            }

            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights_path), DType::F16, &device)
                    .map_err(|e| {
                        CandleAdapterError::ModelNotLoaded(format!(
                            "Failed to load weights from {:?}: {}",
                            weights_path, e
                        ))
                    })?
            };

            // 5. Build model
            tracing::debug!("Building Qwen2 model with config: {:?}", config);

            let model = Qwen2Model::new(&config, vb).map_err(|e| {
                CandleAdapterError::ModelNotLoaded(format!("Failed to build Qwen2 model: {}", e))
            })?;

            tracing::info!(
                "Successfully loaded model: {} (device: {:?})",
                model_name,
                device
            );

            Ok(LoadedModel::new(model, tokenizer, device, model_name))
        })
        .await
        .map_err(|e| {
            CandleAdapterError::InferenceFailed(format!("Task join error during model load: {}", e))
        })?
    }

    /// Clear the model cache (free memory)
    ///
    /// This removes all models from the cache. Subsequent load_model() calls
    /// will reload from disk.
    ///
    /// Useful for:
    /// - Freeing VRAM when switching models
    /// - Forcing a reload after model updates
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
        tracing::info!("Model cache cleared");
    }

    /// Get the number of cached models
    pub async fn cache_size(&self) -> usize {
        let cache = self.cache.read().await;
        cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_model_pool_creation() {
        let pool = ModelPool::new("./models".into());
        assert_eq!(pool.cache_size().await, 0);
    }

    #[tokio::test]
    async fn test_clear_cache() {
        let pool = ModelPool::new("./models".into());
        pool.clear_cache().await;
        assert_eq!(pool.cache_size().await, 0);
    }
}
