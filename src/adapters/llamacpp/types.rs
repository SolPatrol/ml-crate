//! Types for the LlamaCpp Adapter

// Re-export llama-cpp-2 types for convenience
pub use llama_cpp_2::context::params::LlamaContextParams;
pub use llama_cpp_2::context::LlamaContext;
pub use llama_cpp_2::llama_backend::LlamaBackend;
pub use llama_cpp_2::model::params::LlamaModelParams;
pub use llama_cpp_2::model::LlamaModel;

/// LoadedModel - Contains a fully loaded llama.cpp model ready for inference
///
/// # Thread Safety Design
///
/// `LlamaModel` is `Send + Sync` and can be shared across threads.
/// `LlamaContext` is `!Send + !Sync` and must be created per-request.
///
/// This struct holds the model and backend. Contexts are created fresh for each
/// inference request inside `spawn_blocking` to avoid thread safety issues.
///
/// # Usage
///
/// ```ignore
/// let loaded = LoadedModel::load("model.gguf", 32, 2048)?;
/// // Context created per-request in generate_blocking()
/// ```
pub struct LoadedModel {
    /// The llama.cpp backend (must stay alive while model is in use)
    pub backend: LlamaBackend,

    /// The llama.cpp model (Send + Sync, shareable across threads)
    pub model: LlamaModel,

    /// Model name for logging and identification
    pub model_name: String,

    /// Number of GPU layers loaded (for diagnostics)
    pub n_gpu_layers: u32,

    /// Context size for inference
    pub n_ctx: u32,
}

impl LoadedModel {
    /// Load a GGUF model from a file path
    ///
    /// # Arguments
    /// * `path` - Path to the GGUF model file
    /// * `n_gpu_layers` - Number of layers to offload to GPU (0 for CPU-only)
    /// * `n_ctx` - Context length for inference
    ///
    /// # Returns
    /// A LoadedModel ready for inference, or an error if loading fails
    pub fn load(
        path: &str,
        n_gpu_layers: u32,
        n_ctx: u32,
    ) -> Result<Self, crate::adapters::llamacpp::error::LlamaCppError> {
        use crate::adapters::llamacpp::error::LlamaCppError;

        // Initialize the llama.cpp backend
        let backend = LlamaBackend::init()
            .map_err(|e| LlamaCppError::BackendError(format!("Failed to init backend: {e}")))?;

        // Configure model parameters
        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);

        // Load the model from file
        let model = LlamaModel::load_from_file(&backend, path, &model_params)
            .map_err(|e| LlamaCppError::ModelNotLoaded(format!("Failed to load model: {e}")))?;

        // Extract model name from path
        let model_name = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(Self {
            backend,
            model,
            model_name,
            n_gpu_layers,
            n_ctx,
        })
    }

    /// Create a new context for inference
    ///
    /// This should be called inside `spawn_blocking` since LlamaContext
    /// is `!Send + !Sync` and cannot cross thread boundaries.
    pub fn create_context(
        &self,
    ) -> Result<LlamaContext<'_>, crate::adapters::llamacpp::error::LlamaCppError> {
        use crate::adapters::llamacpp::error::LlamaCppError;
        use std::num::NonZeroU32;

        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(self.n_ctx));

        self.model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| LlamaCppError::BackendError(format!("Failed to create context: {e}")))
    }

    /// Get the model name
    pub fn name(&self) -> &str {
        &self.model_name
    }

    /// Get number of GPU layers
    pub fn gpu_layers(&self) -> u32 {
        self.n_gpu_layers
    }

    /// Get context size
    pub fn context_size(&self) -> u32 {
        self.n_ctx
    }
}

// Note: LoadedModel integration tests are in adapter.rs (test_adapter_from_loaded_model, etc.)
// They use a shared OnceLock model to avoid LlamaBackend singleton issues.
// Run with: cargo test --features vulkan adapters::llamacpp -- --ignored
