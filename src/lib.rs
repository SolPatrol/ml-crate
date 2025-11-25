//! # ml-crate-dsrs: DSPy-RS Model Crate
//!
//! GPU-accelerated nano LLM inference for the Omakase Gaming project using dspy-rs
//! for dynamic agent behaviors through optimized prompt programming.
//!
//! ## Architecture
//!
//! This crate provides a Candle-based adapter for dspy-rs, enabling local model
//! inference with the DSPy Module system (Signatures, Predictors, Optimizers).
//!
//! ## Components
//!
//! 1. **Hardware Manager** - VRAM detection and backend selection
//! 2. **Model Pool** - Qwen2.5-0.5B lifecycle management (load, warmup, unload)
//! 3. **Candle Adapter** - dspy-rs Adapter trait implementation
//! 4. **DSPy Engine** - Module pool, execution, and Tool trait for Rhai integration
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use ml_crate_dsrs::adapters::candle::{CandleAdapter, CandleConfig};
//! use ml_crate_dsrs::inference::{DSPyEngine, SignatureRegistry};
//! use ml_crate_dsrs::ModelPool;
//! use dspy_rs::{configure, LM, Predict, Signature, example};
//! use std::path::PathBuf;
//! use std::sync::Arc;
//!
//! #[derive(Signature)]
//! struct QA {
//!     #[input]
//!     question: String,
//!     #[output]
//!     answer: String,
//! }
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // 1. Load model from Model Pool
//!     let model_pool = ModelPool::new("./models".into());
//!     let loaded = model_pool.load_model("Qwen2.5-0.5B").await?;
//!
//!     // 2. Create adapter
//!     let config = CandleConfig::default();
//!     let adapter = CandleAdapter::from_loaded_model(Arc::new(loaded), config);
//!
//!     // 3. Create signature registry and register signatures
//!     let mut registry = SignatureRegistry::new();
//!     registry.register::<QA>("qa");
//!
//!     // 4. Create DSPy Engine
//!     let engine = DSPyEngine::new(
//!         PathBuf::from("./modules"),
//!         Arc::new(adapter),
//!         registry.into_shared(),
//!     ).await?;
//!
//!     // 5. Invoke a module
//!     let result = engine.invoke("qa.simple", serde_json::json!({
//!         "question": "What is Rust?"
//!     })).await?;
//!
//!     Ok(())
//! }
//! ```

// Declare external crate - rig-core (hyphen in Cargo.toml becomes underscore in code)
extern crate rig_core;

pub mod adapters;
pub mod inference;
pub mod model_pool;

// Re-export commonly used types from adapters
pub use adapters::candle::{CandleAdapter, CandleConfig, CandleAdapterError};

// Re-export commonly used types from model_pool
pub use model_pool::ModelPool;

// Re-export commonly used types from inference
pub use inference::{
    DSPyEngine, DSPyEngineError, ModuleManifest, ModuleEntry,
    OptimizedModule, SignatureRegistry,
};
