//! DSPy Engine - Core inference engine for loading and executing pre-optimized DSPy modules
//!
//! This module provides the infrastructure for loading optimized DSPy modules from disk
//! and executing them using the CandleAdapter for local inference.
//!
//! # Architecture
//!
//! - **OptimizedModule**: Represents a pre-optimized DSPy module loaded from JSON
//! - **ModuleManifest**: Registry of available modules with metadata
//! - **SignatureRegistry**: Factory for creating signature instances by name
//! - **DSPyEngine**: Main engine that orchestrates module loading and inference
//!
//! # Example
//!
//! ```rust,ignore
//! use ml_crate_dsrs::inference::{DSPyEngine, SignatureRegistry};
//! use std::path::PathBuf;
//!
//! // Create signature registry and register signatures
//! let mut registry = SignatureRegistry::new();
//! registry.register::<MyQASignature>("qa");
//!
//! // Create engine with modules directory
//! let engine = DSPyEngine::new(
//!     PathBuf::from("./modules"),
//!     adapter,
//!     registry,
//! ).await?;
//!
//! // Invoke a module
//! let result = engine.invoke("qa_module", json!({"question": "What is Rust?"})).await?;
//! ```

pub mod conversion;
pub mod engine;
pub mod error;
pub mod hotreload;
pub mod manifest;
pub mod module;
pub mod registry;

// Re-export commonly used types
pub use conversion::{example_to_value, prediction_to_value, value_to_example};
pub use engine::DSPyEngine;
pub use error::DSPyEngineError;
pub use hotreload::{HotReloadConfig, HotReloadEvent, HotReloadHandle, HotReloadStats};
pub use manifest::{ModuleEntry, ModuleManifest};
pub use module::{Demo, FieldDefinition, ModuleMetadata, OptimizedModule, PredictorType, SignatureDefinition};
pub use registry::SignatureRegistry;
