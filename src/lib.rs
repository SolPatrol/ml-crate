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
//! 2. **Model Pool** - Qwen3-0.6B lifecycle management
//! 3. **Candle Adapter** - dspy-rs Adapter trait implementation (THIS)
//! 4. **Agent Registry** - DSPy Module selection per agent type
//! 5. **Context Builder** - Game state formatting
//! 6. **Tool Registry** - Function calling capabilities
//! 7. **Inference API** - High-level generation interface
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use ml_crate_dsrs::adapters::candle::{CandleAdapter, CandleConfig};
//! use dspy_rs::{configure, LM, Predict, Signature, example};
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
//!     // 1. Load model from Model Pool (Phase 1)
//!     // let model_pool = ModelPool::new("./models".into());
//!     // let loaded = model_pool.load_model("Qwen2.5-0.5B").await?;
//!
//!     // 2. Create adapter
//!     let config = CandleConfig::default(); // Uses 32K context for Qwen2.5-0.5B
//!     // let adapter = CandleAdapter::from_loaded_model(loaded, config);
//!
//!     // 3. Configure dspy-rs
//!     // configure(adapter, None);
//!
//!     // 4. Use predictor
//!     // let qa = Predict::new(QA::new());
//!     // let result = qa.forward(example! {
//!     //     "question": "input" => "What is Rust?"
//!     // }).await?;
//!
//!     Ok(())
//! }
//! ```

// Declare external crate - rig-core (hyphen in Cargo.toml becomes underscore in code)
extern crate rig_core;

pub mod adapters;
pub mod model_pool;

// Re-export commonly used types
pub use adapters::candle::{CandleAdapter, CandleConfig, CandleAdapterError};
pub use model_pool::ModelPool;
