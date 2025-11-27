//! Adapter implementations for dspy-rs
//!
//! This module contains adapter implementations that bridge dspy-rs with
//! various model backends.
//!
//! # Available Adapters
//!
//! - [`llamacpp`] - llama.cpp-based adapter (AMD, NVIDIA, Intel, Apple via Vulkan/CUDA/Metal)

pub mod llamacpp;

pub use llamacpp::LlamaCppAdapter;
