//! Model Pool - Manages loading and lifecycle of LlamaCpp models
//!
//! The Model Pool is responsible for:
//! - Loading GGUF models from disk
//! - GPU layer configuration (Vulkan/CUDA/Metal)
//! - Model caching (avoid reloading)
//!
//! # Example
//!
//! ```rust,ignore
//! use ml_crate_dsrs::model_pool::ModelPool;
//!
//! let pool = ModelPool::new("./models/gguf".into());
//!
//! // Load model (will cache for reuse)
//! let loaded = pool.load_model("qwen2.5-0.5b-instruct-q4_k_m.gguf").await?;
//!
//! // Second load is instant (from cache)
//! let loaded_again = pool.load_model("qwen2.5-0.5b-instruct-q4_k_m.gguf").await?;
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::adapters::llamacpp::{LoadedModel, LlamaCppError};

/// Result type for ModelPool operations
pub type Result<T> = std::result::Result<T, LlamaCppError>;

/// Configuration for ModelPool
#[derive(Clone, Debug)]
pub struct ModelPoolConfig {
    /// Number of GPU layers to offload (99 = all layers)
    pub n_gpu_layers: u32,

    /// Context size for inference
    pub n_ctx: u32,
}

impl Default for ModelPoolConfig {
    fn default() -> Self {
        Self {
            n_gpu_layers: 99, // Offload all layers to GPU by default
            n_ctx: 2048,      // Default context size
        }
    }
}

/// ModelPool manages the lifecycle of loaded LlamaCpp models
///
/// # Example
///
/// ```rust,ignore
/// use ml_crate_dsrs::model_pool::{ModelPool, ModelPoolConfig};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let pool = ModelPool::new("./models/gguf".into(), ModelPoolConfig::default());
///
/// // Load model (will cache for reuse)
/// let loaded = pool.load_model("qwen2.5-0.5b-instruct-q4_k_m.gguf").await?;
///
/// // Second load is instant (from cache)
/// let loaded_again = pool.load_model("qwen2.5-0.5b-instruct-q4_k_m.gguf").await?;
/// # Ok(())
/// # }
/// ```
pub struct ModelPool {
    /// Cache of loaded models (model_name -> LoadedModel)
    cache: Arc<RwLock<HashMap<String, Arc<LoadedModel>>>>,

    /// Base directory where models are stored
    models_dir: PathBuf,

    /// Configuration for model loading
    config: ModelPoolConfig,
}

impl ModelPool {
    /// Create a new ModelPool with default configuration
    ///
    /// # Arguments
    ///
    /// * `models_dir` - Directory containing GGUF model files
    ///
    /// # Example
    ///
    /// ```rust
    /// use ml_crate_dsrs::model_pool::ModelPool;
    ///
    /// let pool = ModelPool::new("./models/gguf".into(), Default::default());
    /// ```
    pub fn new(models_dir: PathBuf, config: ModelPoolConfig) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            models_dir,
            config,
        }
    }

    /// Load a model by name (with caching)
    ///
    /// If the model is already loaded, returns the cached version.
    /// Otherwise, loads from disk and adds to cache.
    ///
    /// # Arguments
    ///
    /// * `model_name` - Name of the GGUF model file (e.g., "qwen2.5-0.5b-instruct-q4_k_m.gguf")
    ///
    /// # Returns
    ///
    /// Arc-wrapped LoadedModel ready for inference
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model file doesn't exist
    /// - Model loading fails
    /// - Backend initialization fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// # use ml_crate_dsrs::model_pool::ModelPool;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let pool = ModelPool::new("./models/gguf".into(), Default::default());
    /// let loaded = pool.load_model("qwen2.5-0.5b-instruct-q4_k_m.gguf").await?;
    ///
    /// println!("Loaded model: {}", loaded.name());
    /// println!("GPU layers: {}", loaded.gpu_layers());
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
        let model_path_str = model_path.to_string_lossy().to_string();

        // Check if file exists
        if !model_path.exists() {
            return Err(LlamaCppError::ModelNotLoaded(format!(
                "Model file not found: {}",
                model_path_str
            )));
        }

        let n_gpu_layers = self.config.n_gpu_layers;
        let n_ctx = self.config.n_ctx;

        // Load model in blocking task (CPU/GPU-bound work)
        let loaded = tokio::task::spawn_blocking(move || {
            LoadedModel::load(&model_path_str, n_gpu_layers, n_ctx)
        })
        .await
        .map_err(|e| LlamaCppError::InferenceFailed(format!("Task join error: {}", e)))??;

        let loaded = Arc::new(loaded);

        // Add to cache
        {
            let mut cache = self.cache.write().await;
            cache.insert(model_name.to_string(), Arc::clone(&loaded));
        }

        tracing::info!(
            "Model loaded and cached: {} (GPU layers: {}, context: {})",
            model_name,
            loaded.gpu_layers(),
            loaded.context_size()
        );
        Ok(loaded)
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

    /// Get the models directory
    pub fn models_dir(&self) -> &PathBuf {
        &self.models_dir
    }

    /// Get the configuration
    pub fn config(&self) -> &ModelPoolConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_model_pool_creation() {
        let pool = ModelPool::new("./models/gguf".into(), ModelPoolConfig::default());
        assert_eq!(pool.cache_size().await, 0);
    }

    #[tokio::test]
    async fn test_clear_cache() {
        let pool = ModelPool::new("./models/gguf".into(), ModelPoolConfig::default());
        pool.clear_cache().await;
        assert_eq!(pool.cache_size().await, 0);
    }

    #[tokio::test]
    async fn test_config_defaults() {
        let config = ModelPoolConfig::default();
        assert_eq!(config.n_gpu_layers, 99);
        assert_eq!(config.n_ctx, 2048);
    }

    #[tokio::test]
    async fn test_model_not_found() {
        let pool = ModelPool::new("./models/gguf".into(), ModelPoolConfig::default());
        let result = pool.load_model("nonexistent-model.gguf").await;
        assert!(result.is_err());
        if let Err(LlamaCppError::ModelNotLoaded(msg)) = result {
            assert!(msg.contains("not found"));
        } else {
            panic!("Expected ModelNotLoaded error");
        }
    }

    #[tokio::test]
    #[ignore = "requires GGUF model file"]
    async fn test_load_real_model() {
        let pool = ModelPool::new("./models/gguf".into(), ModelPoolConfig::default());
        let loaded = pool
            .load_model("qwen2.5-0.5b-instruct-q4_k_m.gguf")
            .await
            .expect("Model load failed");

        assert!(loaded.gpu_layers() > 0);
        assert_eq!(loaded.context_size(), 2048);
        assert_eq!(pool.cache_size().await, 1);

        // Second load should be cached
        let loaded2 = pool
            .load_model("qwen2.5-0.5b-instruct-q4_k_m.gguf")
            .await
            .expect("Cached load failed");
        assert_eq!(pool.cache_size().await, 1); // Still just 1 model

        // Should be same Arc
        assert!(Arc::ptr_eq(&loaded, &loaded2));
    }
}
