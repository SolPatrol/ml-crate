//! LlamaCpp Adapter module
//!
//! This module provides llama.cpp-based LLM inference for the ml-crate-dsrs system.
//! It implements the dspy-rs `Adapter` trait using llama-cpp-2 Rust bindings.
//!
//! # Backend Support
//!
//! - **Vulkan** (default) - AMD, NVIDIA, Intel GPUs
//! - **CUDA** - NVIDIA GPUs (+10-20% vs Vulkan)
//! - **Metal** - Apple Silicon
//! - **CPU** - Fallback (always available)
//!
//! # Example
//!
//! ```rust,ignore
//! use ml_crate_dsrs::adapters::llamacpp::{LlamaCppAdapter, LlamaCppConfig, LoadedModel};
//! use std::sync::Arc;
//!
//! // Model Pool loads the GGUF model
//! let loaded_model = model_pool.load_model("qwen2.5-0.5b-q4_k_m").await?;
//!
//! // Create adapter with configuration
//! let adapter = LlamaCppAdapter::from_loaded_model(
//!     Arc::new(loaded_model),
//!     LlamaCppConfig::default()
//! );
//! ```
//!
//! # Module Structure
//!
//! - `adapter` - LlamaCppAdapter implementation
//! - `config` - Configuration struct
//! - `error` - Error types
//! - `types` - LoadedModel and related types

mod adapter;
mod config;
mod error;
mod types;

pub use adapter::LlamaCppAdapter;
pub use config::LlamaCppConfig;
pub use error::{LlamaCppError, Result};
pub use types::LoadedModel;

// Re-export dspy-rs types for convenience (matches candle module pattern)
pub use dspy_rs::{Chat, Example, LM, LmUsage, Message, MetaSignature, Prediction};
