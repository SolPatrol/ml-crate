//! Adapter implementations for dspy-rs
//!
//! This module contains adapter implementations that bridge dspy-rs with
//! various model backends.

pub mod candle;

pub use candle::CandleAdapter;
