//! Candle Adapter for dspy-rs
//!
//! This module implements the dspy-rs `Adapter` trait to provide Candle-based
//! LLM inference for the ml-crate-dsrs system.
//!
//! ## Architecture
//!
//! ```text
//! DSPy Predictor (Predict, ChainOfThought, ReAct)
//!     ↓ uses
//! Adapter trait (interface)
//!     ↓ implemented by
//! CandleAdapter (our implementation)
//!     ↓ receives
//! LoadedModel from Model Pool
//!     ↓ uses
//! Candle Model (Qwen3-0.6B)
//! ```
//!
//! ## Key Points
//!
//! - **Implements ONLY the `Adapter` trait** from dspy-rs v0.7.3
//! - **Receives pre-loaded models** from Model Pool (does NOT load models)
//! - **Three methods**: `format()`, `parse_response()`, `call()`
//! - **Works with LM struct** for configuration
//!
//! ## Example
//!
//! ```rust,ignore
//! use ml_crate_dsrs::adapters::candle::{CandleAdapter, CandleConfig};
//! use dspy_rs::{configure, Predict, Signature, example};
//!
//! #[derive(Signature)]
//! struct QA {
//!     #[input]
//!     question: String,
//!     #[output]
//!     answer: String,
//! }
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create adapter (receives model from Model Pool)
//! // let model_pool = ModelPool::new("./models".into());
//! // let loaded = model_pool.load_model("Qwen2.5-0.5B").await?;
//! // let adapter = CandleAdapter::from_loaded_model(loaded, CandleConfig::default());
//!
//! // Configure dspy-rs
//! // configure(adapter, None);
//!
//! // Use predictor
//! // let qa = Predict::new(QA::new());
//! // let result = qa.forward(example! {
//! //     "question": "input" => "What is Rust?"
//! // }).await?;
//! # Ok(())
//! # }
//! ```

pub mod adapter;
pub mod config;
pub mod error;

pub use adapter::{CandleAdapter, LoadedModel};
pub use config::CandleConfig;
pub use error::{CandleAdapterError, Result};

// Re-export dspy-rs types for convenience
pub use dspy_rs::{Chat, Example, LM, LmUsage, Message, MetaSignature, Prediction};
