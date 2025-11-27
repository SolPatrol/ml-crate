//! Adapter implementations for dspy-rs
//!
//! This module contains adapter implementations that bridge dspy-rs with
//! various model backends.
//!
//! # Available Adapters
//!
//! - [`candle`] - Candle-based adapter (NVIDIA CUDA, Apple Metal)
//! - [`llamacpp`] - llama.cpp-based adapter (AMD, NVIDIA, Intel, Apple via Vulkan/CUDA/Metal)

pub mod candle;
pub mod llamacpp;

pub use candle::CandleAdapter;
pub use llamacpp::LlamaCppAdapter;
