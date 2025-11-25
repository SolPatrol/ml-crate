//! DSPyEngine - Core orchestrator for loading and executing pre-optimized DSPy modules
//!
//! The DSPyEngine is responsible for:
//! - Loading optimized modules from disk
//! - Managing the signature registry
//! - Executing inference with Predict or ChainOfThought predictors
//! - Converting between JSON values and dspy-rs types
//!
//! # Architecture
//!
//! ```text
//! JSON Input (Value)
//!     ↓
//! DSPyEngine.invoke(module_id, input)
//!     ↓
//! OptimizedModule (loaded from disk)
//!     ↓
//! SignatureRegistry.create(signature_name)
//!     ↓
//! Configure signature with demos + instruction
//!     ↓
//! CandleAdapter.call() via dspy-rs Predictor
//!     ↓
//! JSON Output (Value)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use ml_crate_dsrs::inference::{DSPyEngine, SignatureRegistry};
//! use serde_json::json;
//!
//! // Consumer registers their signatures
//! let mut registry = SignatureRegistry::new();
//! registry.register::<NPCDialogue>("npc.dialogue");
//!
//! // Create engine
//! let engine = DSPyEngine::new(
//!     PathBuf::from("./modules"),
//!     adapter,
//!     registry.into_shared(),
//! ).await?;
//!
//! // Invoke a module
//! let result = engine.invoke("npc.dialogue.casual", json!({
//!     "npc_personality": "gruff blacksmith",
//!     "player_message": "Hello there!"
//! })).await?;
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use dspy_rs::{Example, MetaSignature, Predict, Predictor};
use serde_json::Value;
use tokio::sync::RwLock;

use crate::adapters::candle::CandleAdapter;

use super::conversion::{demos_to_examples, prediction_to_value, value_to_example};
use super::error::{DSPyEngineError, Result};
use super::hotreload::{HotReloadConfig, HotReloadHandle, HotReloadManager};
use super::manifest::{load_json_with_context, ModuleManifest};
use super::module::{OptimizedModule, PredictorType};
use super::registry::SignatureRegistry;

/// DSPyEngine - Main orchestrator for module execution
///
/// The engine manages the lifecycle of optimized modules and provides
/// the interface for executing inference.
pub struct DSPyEngine {
    /// Loaded modules indexed by module_id
    modules: Arc<RwLock<HashMap<String, OptimizedModule>>>,

    /// Module manifest for module metadata and paths
    manifest: Arc<RwLock<ModuleManifest>>,

    /// Directory containing module files
    modules_dir: PathBuf,

    /// CandleAdapter for local inference
    adapter: Arc<CandleAdapter>,

    /// Signature registry (provided by consumer)
    signature_registry: Arc<SignatureRegistry>,

    /// Flag indicating if dspy-rs has been configured
    configured: bool,
}

impl DSPyEngine {
    /// Create a new DSPyEngine
    ///
    /// # Arguments
    ///
    /// * `modules_dir` - Path to directory containing module JSON files and manifest.json
    /// * `adapter` - CandleAdapter instance for local GPU inference
    /// * `signature_registry` - Consumer-provided registry mapping signature names to types
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Consumer creates and populates the signature registry
    /// let mut registry = SignatureRegistry::new();
    /// registry.register::<NPCDialogue>("npc.dialogue");
    ///
    /// // Pass registry to engine
    /// let engine = DSPyEngine::new(
    ///     PathBuf::from("./modules"),
    ///     adapter,
    ///     Arc::new(registry),
    /// ).await?;
    /// ```
    pub async fn new(
        modules_dir: PathBuf,
        adapter: Arc<CandleAdapter>,
        signature_registry: Arc<SignatureRegistry>,
    ) -> Result<Self> {
        let engine = Self {
            modules: Arc::new(RwLock::new(HashMap::new())),
            manifest: Arc::new(RwLock::new(ModuleManifest::default())),
            modules_dir,
            adapter,
            signature_registry,
            configured: false,
        };

        // Load manifest and all modules
        engine.reload_all().await?;

        Ok(engine)
    }

    /// Create engine without loading modules (for testing)
    pub fn new_empty(
        modules_dir: PathBuf,
        adapter: Arc<CandleAdapter>,
        signature_registry: Arc<SignatureRegistry>,
    ) -> Self {
        Self {
            modules: Arc::new(RwLock::new(HashMap::new())),
            manifest: Arc::new(RwLock::new(ModuleManifest::default())),
            modules_dir,
            adapter,
            signature_registry,
            configured: false,
        }
    }

    /// Configure dspy-rs global settings
    ///
    /// This must be called before any predictor operations.
    /// It's called automatically on first invoke() if not already configured.
    ///
    /// Note: dspy-rs uses global state, so this affects all predictors.
    pub async fn configure_dspy(&mut self) -> Result<()> {
        if self.configured {
            return Ok(());
        }

        // Create LM configured for local inference
        // dspy-rs requires an LM struct, but our CandleAdapter handles all inference
        // locally using the embedded Candle model - no external API calls are made
        let lm = dspy_rs::LM::builder()
            .model("local-candle".to_string())
            .base_url("http://localhost:0".to_string()) // Local mode - CandleAdapter handles inference
            .build()
            .await
            .map_err(|e| DSPyEngineError::RuntimeError(format!("Failed to create LM: {}", e)))?;

        // Configure global settings with LM and our CandleAdapter
        // Note: CandleAdapter implements Clone via derived Clone on the struct
        // The adapter is cloned here because configure() takes ownership
        dspy_rs::configure(lm, (*self.adapter).clone());

        self.configured = true;
        tracing::info!("DSPy global settings configured with CandleAdapter");

        Ok(())
    }

    /// Load/reload all modules from disk
    pub async fn reload_all(&self) -> Result<()> {
        let manifest_path = self.modules_dir.join("manifest.json");

        // Load manifest (or use empty if not found)
        let manifest: ModuleManifest = if manifest_path.exists() {
            load_json_with_context(&manifest_path, "Loading manifest")?
        } else {
            tracing::warn!(
                "No manifest.json found in {:?}, using empty manifest",
                self.modules_dir
            );
            ModuleManifest::default()
        };

        // Load all modules
        let mut modules = self.modules.write().await;
        modules.clear();

        for (id, entry) in &manifest.modules {
            let module_path = self.modules_dir.join(&entry.path);
            match load_json_with_context::<OptimizedModule>(
                &module_path,
                &format!("Loading module '{}'", id),
            ) {
                Ok(module) => {
                    tracing::debug!("Loaded module: {}", id);
                    modules.insert(id.clone(), module);
                }
                Err(e) => {
                    tracing::error!("Failed to load module '{}': {}", id, e);
                    // Continue loading other modules
                }
            }
        }

        *self.manifest.write().await = manifest;

        tracing::info!("Loaded {} modules from {:?}", modules.len(), self.modules_dir);

        Ok(())
    }

    /// Reload a single module by ID
    pub async fn reload_module(&self, module_id: &str) -> Result<()> {
        let manifest = self.manifest.read().await;
        let entry = manifest
            .modules
            .get(module_id)
            .ok_or_else(|| DSPyEngineError::module_not_found(module_id))?;

        let module_path = self.modules_dir.join(&entry.path);
        let module: OptimizedModule = load_json_with_context(
            &module_path,
            &format!("Reloading module '{}'", module_id),
        )?;

        self.modules
            .write()
            .await
            .insert(module_id.to_string(), module);

        tracing::debug!("Reloaded module: {}", module_id);

        Ok(())
    }

    /// Get a module by ID (clone)
    pub async fn get_module(&self, module_id: &str) -> Option<OptimizedModule> {
        self.modules.read().await.get(module_id).cloned()
    }

    /// Check if a module exists
    pub async fn has_module(&self, module_id: &str) -> bool {
        self.modules.read().await.contains_key(module_id)
    }

    /// Get all loaded module IDs
    pub async fn module_ids(&self) -> Vec<String> {
        self.modules.read().await.keys().cloned().collect()
    }

    /// Get the number of loaded modules
    pub async fn module_count(&self) -> usize {
        self.modules.read().await.len()
    }

    /// Get the signature registry
    pub fn signature_registry(&self) -> &Arc<SignatureRegistry> {
        &self.signature_registry
    }

    /// Invoke a module (async)
    ///
    /// This is the main entry point for executing inference.
    ///
    /// # Arguments
    ///
    /// * `module_id` - The ID of the module to invoke
    /// * `input` - JSON input containing field values
    ///
    /// # Returns
    ///
    /// JSON output containing the predicted field values.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = engine.invoke("npc.dialogue.casual", json!({
    ///     "npc_personality": "gruff blacksmith",
    ///     "player_message": "Hello!"
    /// })).await?;
    ///
    /// let response = result["response"].as_str().unwrap();
    /// ```
    pub async fn invoke(&self, module_id: &str, input: Value) -> Result<Value> {
        let module = self
            .get_module(module_id)
            .await
            .ok_or_else(|| DSPyEngineError::module_not_found(module_id))?;

        // Dispatch based on predictor type
        match module.predictor_type {
            PredictorType::Predict => self.execute_predict(&module, input).await,
            PredictorType::ChainOfThought => self.execute_chain_of_thought(&module, input).await,
        }
    }

    /// Invoke a module synchronously (blocking)
    ///
    /// This is useful for integration with synchronous code like Rhai.
    /// Creates a new runtime for the blocking call.
    pub fn invoke_sync(&self, module_id: &str, input: Value) -> Result<Value> {
        // Use tokio's current runtime or create a new one
        let input_clone = input.clone();
        let rt = tokio::runtime::Handle::try_current()
            .map(|h| h.block_on(self.invoke(module_id, input)))
            .unwrap_or_else(|_| {
                // No runtime, create a new one
                let rt = tokio::runtime::Runtime::new().map_err(|e| {
                    DSPyEngineError::RuntimeError(format!("Failed to create runtime: {}", e))
                })?;
                rt.block_on(self.invoke(module_id, input_clone))
            });
        rt
    }

    /// Execute with Predict predictor
    async fn execute_predict(&self, module: &OptimizedModule, input: Value) -> Result<Value> {
        // 1. Get signature from registry
        let mut signature = self
            .signature_registry
            .create(&module.signature_name)
            .ok_or_else(|| DSPyEngineError::signature_not_found(&module.signature_name))?;

        // 2. Configure signature with module's optimized demos and instruction
        self.configure_signature(&mut signature, module)?;

        // 3. Convert input Value to Example
        let example = value_to_example(&input, signature.as_ref());

        // 4. Create predictor and run forward pass
        let predictor = Predict::new(SignatureWrapper(signature));
        let prediction = predictor
            .forward(example)
            .await
            .map_err(|e| DSPyEngineError::inference(e.to_string()))?;

        // 5. Convert Prediction to Value
        Ok(prediction_to_value(&prediction))
    }

    /// Execute with ChainOfThought predictor
    async fn execute_chain_of_thought(
        &self,
        module: &OptimizedModule,
        input: Value,
    ) -> Result<Value> {
        // 1. Get signature from registry
        let mut signature = self
            .signature_registry
            .create(&module.signature_name)
            .ok_or_else(|| DSPyEngineError::signature_not_found(&module.signature_name))?;

        // 2. Configure signature with module's optimized demos and instruction
        self.configure_signature(&mut signature, module)?;

        // 3. Convert input Value to Example
        let example = value_to_example(&input, signature.as_ref());

        // 4. Create predictor and run forward pass
        // Note: ChainOfThought automatically adds "reasoning" field to outputs
        let predictor = Predict::new(SignatureWrapper(signature));
        // For now, we use Predict for both - ChainOfThought support requires
        // modifying the signature to include reasoning field
        // TODO: Implement proper ChainOfThought when dspy-rs exposes it
        let prediction = predictor
            .forward(example)
            .await
            .map_err(|e| DSPyEngineError::inference(e.to_string()))?;

        // 5. Convert Prediction to Value
        Ok(prediction_to_value(&prediction))
    }

    /// Configure a signature with module's optimized demos and instruction
    fn configure_signature(
        &self,
        signature: &mut Box<dyn MetaSignature>,
        module: &OptimizedModule,
    ) -> Result<()> {
        // Set demonstrations
        if !module.demos.is_empty() {
            let examples = demos_to_examples(&module.demos);
            signature
                .set_demos(examples)
                .map_err(|e| DSPyEngineError::runtime(format!("Failed to set demos: {}", e)))?;
        }

        // Set optimized instruction
        if !module.instruction.is_empty() {
            signature
                .update_instruction(module.instruction.clone())
                .map_err(|e| {
                    DSPyEngineError::runtime(format!("Failed to update instruction: {}", e))
                })?;
        }

        Ok(())
    }

    /// Add a module programmatically (for testing)
    pub async fn add_module(&self, module: OptimizedModule) {
        self.modules
            .write()
            .await
            .insert(module.module_id.clone(), module);
    }

    /// Remove a module
    pub async fn remove_module(&self, module_id: &str) -> Option<OptimizedModule> {
        self.modules.write().await.remove(module_id)
    }

    /// Enable hot reload for automatic module reloading
    ///
    /// Starts a file watcher on the modules directory that automatically
    /// reloads modules when their JSON files change.
    ///
    /// # Arguments
    ///
    /// * `config` - Hot reload configuration
    ///
    /// # Returns
    ///
    /// A `HotReloadHandle` that provides access to events and allows stopping the watcher.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Enable with default config
    /// let hot_reload = engine.enable_hot_reload(HotReloadConfig::default())?;
    ///
    /// // Monitor events
    /// tokio::spawn(async move {
    ///     while let Some(event) = hot_reload.events().recv().await {
    ///         println!("Hot reload event: {:?}", event);
    ///     }
    /// });
    /// ```
    pub fn enable_hot_reload(&self, config: HotReloadConfig) -> Result<HotReloadHandle> {
        let mut manager = HotReloadManager::new(
            Arc::clone(&self.modules),
            Arc::clone(&self.manifest),
            self.modules_dir.clone(),
            config,
        );

        let events = manager.start()?;

        Ok(HotReloadHandle::new(manager, events))
    }

    /// Get the modules directory path
    pub fn modules_dir(&self) -> &PathBuf {
        &self.modules_dir
    }
}

/// Wrapper to make Box<dyn MetaSignature> work with Predict::new()
///
/// Predict::new() requires `impl MetaSignature`, but we have `Box<dyn MetaSignature>`.
/// This wrapper implements MetaSignature by delegating to the inner boxed trait object.
struct SignatureWrapper(Box<dyn MetaSignature>);

impl MetaSignature for SignatureWrapper {
    fn demos(&self) -> Vec<Example> {
        self.0.demos()
    }

    fn set_demos(&mut self, demos: Vec<Example>) -> anyhow::Result<()> {
        self.0.set_demos(demos)
    }

    fn instruction(&self) -> String {
        self.0.instruction()
    }

    fn input_fields(&self) -> Value {
        self.0.input_fields()
    }

    fn output_fields(&self) -> Value {
        self.0.output_fields()
    }

    fn update_instruction(&mut self, instruction: String) -> anyhow::Result<()> {
        self.0.update_instruction(instruction)
    }

    fn append(&mut self, name: &str, value: Value) -> anyhow::Result<()> {
        self.0.append(name, value)
    }
}

// SignatureWrapper needs to be Send + Sync for async use
// This is safe because Box<dyn MetaSignature> requires Send + Sync
unsafe impl Send for SignatureWrapper {}
unsafe impl Sync for SignatureWrapper {}

impl std::fmt::Debug for DSPyEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DSPyEngine")
            .field("modules_dir", &self.modules_dir)
            .field("configured", &self.configured)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::module::ModuleMetadata;

    // Note: Most tests require a real CandleAdapter with a loaded model,
    // which is covered in integration tests. Here we test the non-inference parts.

    #[test]
    fn test_signature_wrapper_delegates() {
        struct TestSignature {
            instruction: String,
        }

        impl MetaSignature for TestSignature {
            fn demos(&self) -> Vec<Example> {
                vec![]
            }
            fn set_demos(&mut self, _demos: Vec<Example>) -> anyhow::Result<()> {
                Ok(())
            }
            fn instruction(&self) -> String {
                self.instruction.clone()
            }
            fn input_fields(&self) -> Value {
                serde_json::json!(["input"])
            }
            fn output_fields(&self) -> Value {
                serde_json::json!(["output"])
            }
            fn update_instruction(&mut self, instruction: String) -> anyhow::Result<()> {
                self.instruction = instruction;
                Ok(())
            }
            fn append(&mut self, _name: &str, _value: Value) -> anyhow::Result<()> {
                Ok(())
            }
        }

        let inner = TestSignature {
            instruction: "Test".to_string(),
        };
        let wrapper = SignatureWrapper(Box::new(inner));

        assert_eq!(wrapper.instruction(), "Test");
        assert_eq!(wrapper.input_fields(), serde_json::json!(["input"]));
    }

    #[test]
    fn test_optimized_module_creation() {
        let module = OptimizedModule::new(
            "test.module",
            "test.signature",
            "Test instruction",
        )
        .with_predictor_type(PredictorType::ChainOfThought)
        .with_metadata(ModuleMetadata::manual("1.0.0"));

        assert_eq!(module.module_id, "test.module");
        assert_eq!(module.signature_name, "test.signature");
        assert!(module.is_chain_of_thought());
    }
}
