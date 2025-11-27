# LlamaCpp Adapter Specification

**Version**: 1.0.0 (COMPLETE)
**Status**: ✅ MIGRATION COMPLETE
**Dependencies**: dspy-rs (v0.7.3+), Model Pool (Component #2), llama-cpp-2 (Rust bindings)
**Last Updated**: 2025-11-26

---

## Implementation Status

**Phase 0 (Specification)**: ✅ COMPLETE
- [x] Specification document complete
- [x] Architecture validated against dspy-rs (via dspy-researcher)
- [x] Cross-validated against CandleAdapter codebase
- [x] Migration checklist defined

**Phase 1 (Dependencies & Core Types)**: ✅ COMPLETE
- [x] Add `llama-cpp-2` dependency with feature flags (vulkan default)
- [x] Create `src/adapters/llamacpp/` module structure
- [x] Add `LlamaCppError` enum (10 variants, matches CandleAdapterError + BackendError)
- [x] Add `LlamaCppConfig` struct (15 fields, 11 builder methods)
- [x] Implement `LoadedModel` struct (mirrors CandleAdapter)
- [x] Create `LlamaCppAdapter` stub (constructor + getters)
- [x] Verify builds: Vulkan ✅, CUDA ✅, CPU ✅
- [x] Document Windows build setup ([06-windows-build-setup.md](06-windows-build-setup.md))

**Phase 2 (Adapter Implementation)**: ✅ COMPLETE
- [x] Implement `LlamaCppAdapter` with dspy-rs `Adapter` trait
- [x] Implement `from_loaded_model()` constructor
- [x] Port `format()` from CandleAdapter
- [x] Port `format_demonstrations()` for few-shot learning
- [x] Port `parse_response()` with 3-strategy parsing
- [x] Implement `chat_to_prompt()` helper
- [x] Implement `generate()` with `spawn_blocking` pattern
- [x] Implement `generate_with_retry()` for error recovery
- [x] Implement full `call()` method (format → generate → parse)
- [ ] Implement `generate_stream()` for streaming output (deferred to Phase 2B)

**Phase 3 (Real llama-cpp-2 Integration)**: ✅ COMPLETE
- [x] Replace placeholder types with real llama-cpp-2 types
- [x] Implement `LoadedModel::load()` for GGUF model loading
- [x] Implement `generate_blocking()` with full inference pipeline
- [x] Thread safety: context created per-request (solves `!Send + !Sync`)
- [x] Add `seed` config parameter for reproducible sampling
- [x] Fix Windows CRT mismatch (esaxx-rs vs llama-cpp-sys-2)
- [x] Verify GPU inference (Vulkan on NVIDIA GTX 1070)

**Phase 4 (Testing & Validation)**: ✅ COMPLETE
- [x] Unit tests for LlamaCppAdapter (23 tests: 14 unit + 9 integration)
- [x] Integration tests with real GGUF model (all 9 pass with shared model)
- [x] Verify token counts are accurate (prompt + completion tokens tracked)
- [x] Edge case unit tests (6 tests for empty/whitespace/unicode/special chars)
- [x] Verify dspy-rs `configure()` works correctly (`test_dspy_configure()` added)
- [x] Shared model via OnceLock (mirrors production pattern)
- [x] Clippy clean (0 warnings)
- [x] Backend verification: Vulkan ✅, CPU ✅

**Phase 5 (Quality Gates)**: ✅ COMPLETE
- [x] All unit tests pass (23 tests: 14 unit + 9 integration)
- [x] Integration tests with real GGUF model pass
- [x] Clean `cargo clippy` output (0 warnings)
- [x] No panics or unwraps in production code (audit complete)
- [x] Performance benchmarking: ~5.6s GPU vs ~16s CPU (3x speedup)

**Phase 6 (Model Pool Integration & Cleanup)**: ✅ COMPLETE
- [x] Download Qwen GGUF model (qwen2.5-0.5b-instruct-q4_k_m.gguf)
- [x] Update `ModelPool` to load GGUF models via LlamaCpp
- [x] Return `Arc<LoadedModel>` for adapter creation
- [x] Configure GPU layers via `ModelPoolConfig` (n_gpu_layers, n_ctx)
- [x] ModelPool cache works (second load is instant from cache)
- [x] Update DSPyEngine to use LlamaCppAdapter
- [x] Update all documentation (lib.rs, adapters/mod.rs updated)
- [x] Remove Candle dependencies (candle-core, candle-nn, candle-transformers, tokenizers removed)

---

## Validation Report (Phase 0)

### dspy-rs Alignment (via dspy-researcher)

| Component | Status | Notes |
|-----------|--------|-------|
| Adapter trait signature | ✅ Perfect match | Exact match with dspy-rs v0.7.3 |
| Core type usage | ✅ All correct | Chat, Message, Example, Prediction, LmUsage |
| Integration pattern | ✅ Correct | `configure(lm, adapter)` verified |
| Thread safety | ✅ Appropriate | `Arc<Mutex<Model>>` correct for thread-safe access |
| spawn_blocking pattern | ✅ Best practice | Correct approach for CPU/GPU-bound inference |

### Cross-validation Against CandleAdapter Codebase

| Feature | CandleAdapter | LlamaCpp Spec | Status |
|---------|---------------|---------------|--------|
| `LoadedModel` struct | ✅ | ✅ | Pattern matches |
| `from_loaded_model()` | ✅ | ✅ | Identical signature |
| `chat_to_prompt()` | ✅ | ✅ | Identical implementation |
| `format()` with demos | ✅ | ✅ | 3-strategy parsing |
| `parse_response()` | ✅ | ✅ | Field marker + JSON + single-field |
| `call()` flow | ✅ | ✅ | Format → Generate → Parse |
| `generate()` pattern | ✅ | ✅ | spawn_blocking with Arc clone |
| `generate_stream()` | ✅ | ✅ | Streaming support included |
| Error variants | ✅ | ✅ | All CandleAdapterError variants |
| Config fields | ✅ | ✅ | All CandleConfig fields + repeat_penalty |

**Validation Date**: 2025-11-26
**Validated By**: dspy-researcher subagent + manual CandleAdapter code review

---

## Table of Contents

- [Overview](#overview)
- [Purpose](#purpose)
- [Architecture](#architecture)
- [Backend Support](#backend-support)
- [Core Components](#core-components)
- [Error Handling](#error-handling)
- [Configuration](#configuration)
- [Testing Strategy](#testing-strategy)
- [Performance Considerations](#performance-considerations)
- [Dependencies](#dependencies)
- [API Examples](#api-examples)
- [Success Criteria](#success-criteria)
- [Migration from CandleAdapter](#migration-from-candleadapter)
- [Future Enhancements](#future-enhancements)
- [Design Decisions](#design-decisions)
- [References](#references)

---

## Overview

The `llamacpp` adapter module **implements the dspy-rs `Adapter` trait** to provide llama.cpp-based LLM inference for the ml-crate-dsrs system. It acts as the bridge between dspy-rs's abstract adapter requirements and concrete llama.cpp model deployments.

**Module Path**: `ml_crate_dsrs::adapters::llamacpp`
**Parent Crate**: `ml-crate-dsrs` (see [ARCHv2.md](ARCHv2.md))
**Related Components**: Model Pool (component #2), Inference API (component #6)
**Replaces**: `ml_crate_dsrs::adapters::candle` (see [01-candle-adapter.md](01-candle-adapter.md))

---

## Purpose

### What It Does
- Implements dspy-rs's `Adapter` trait for llama.cpp inference
- Provides direct llama.cpp model inference integration via `llama-cpp-2` bindings
- Handles tokenization and text generation (built into llama.cpp)
- Returns Prediction outputs for dspy-rs Modules to use
- Manages inference lifecycle and error handling
- Supports conversation history via Chat/Message types
- Works with `LM` struct for configuration
- **Supports all GPU vendors** via Vulkan (default), CUDA, Metal, or CPU backends

### What It Does NOT Do
- Model lifecycle management (that's Model Pool)
- Model optimization or fine-tuning
- Tool execution (that's Tool Registry)
- Tool call parsing (that's dspy-rs Modules - ReAct/ChainOfThought)
- Agent logic or context building (that's Agent Registry + Context Builder)
- Define Adapter trait (uses dspy-rs provided trait)
- Backend selection logic (that's Hardware Manager)

---

## Architecture

### The dspy-rs Architecture (v0.7.3)

```
DSPy Predictor (Predict, ChainOfThought, ReAct)
    ↓ uses
Adapter trait (interface)
    ↓ implemented by
LlamaCppAdapter (our implementation)
    ↓ receives
LM struct (configuration via Arc<LM>)
    ↓ LlamaCppAdapter calls
llama.cpp (via llama-cpp-2 bindings)
    ↓ executes on
GPU (Vulkan/CUDA/Metal) or CPU
```

**Key Points:**
- `Adapter` trait is the ONLY trait we implement
- `LM` struct is passed to the `call()` method for configuration
- There is NO `LanguageModel` trait in dspy-rs v0.7.3
- Same pattern as CandleAdapter - only the inference engine changes

### Within ml-crate-dsrs

```
┌─────────────────────────────────────────────────────────┐
│                    ml-crate-dsrs                        │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Hardware Manager (Component #1)                 │  │
│  │  • Detects available hardware                    │  │
│  │  • Selects backend: Vulkan/CUDA/Metal/CPU        │  │
│  └────────────────────┬─────────────────────────────┘  │
│                       │ informs                        │
│                       ↓                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Model Pool (Component #2)                       │  │
│  │  • Loads GGUF model via llama.cpp                │  │
│  │  • Configures GPU layers                         │  │
│  │  • Returns Arc<LoadedModel>                      │  │
│  └────────────────────┬─────────────────────────────┘  │
│                       │ provides                       │
│                       │ Arc<LoadedModel>               │
│                       ↓                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │  llamacpp adapter module                         │  │
│  │                                                  │  │
│  │  ┌────────────────────────────────────┐         │  │
│  │  │  LlamaCppAdapter                   │         │  │
│  │  │  from_loaded_model(loaded)         │         │  │
│  │  │  implements Adapter trait          │         │  │
│  │  └────────────────────────────────────┘         │  │
│  └────────────────────┬─────────────────────────────┘  │
│                       │ wrapped by                     │
│                       ↓                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │  LM struct (from dspy-rs)                        │  │
│  │  Configuration & builder for Adapter             │  │
│  └────────────────────┬─────────────────────────────┘  │
│                       │ passed to                      │
│                       ↓                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Inference API (Component #6)                    │  │
│  │  + Agent Registry (Component #3)                 │  │
│  │  Uses Adapter trait via configure()              │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Key Integration Points:**
- `Hardware Manager` detects hardware and selects backend (Vulkan/CUDA/Metal/CPU)
- `Model Pool` **loads and manages** GGUF model with appropriate GPU layers
- `Model Pool` provides `Arc<LoadedModel>` to adapter
- `LlamaCppAdapter` **wraps** loaded model and implements `Adapter` trait
- `LM` struct provides configuration (model name, temp, max_tokens, etc.)
- `Inference API` uses `Adapter` trait via `configure()` to set up global LM
- `Agent Registry` uses configured Adapter for DSPy Module execution
- `dspy-rs Modules` (ReAct/ChainOfThought) handle tool parsing from Prediction outputs

---

## Backend Support

```
llama.cpp (via llama-cpp-2 bindings)
    │
    ├── Vulkan  → AMD, NVIDIA, Intel GPUs (DEFAULT)
    ├── CUDA    → NVIDIA GPUs (+10-20% vs Vulkan)
    ├── Metal   → Apple Silicon
    └── CPU     → Fallback (AVX2/AVX512)
```

| Backend | NVIDIA | AMD | Intel | Apple | Default |
|---------|--------|-----|-------|-------|---------|
| Vulkan | ✅ | ✅ | ✅ | ❌ | ✅ **Yes** |
| CUDA | ✅ | ❌ | ❌ | ❌ | No |
| Metal | ❌ | ❌ | ❌ | ✅ | No |
| CPU | ✅ | ✅ | ✅ | ✅ | Fallback |

**Why Vulkan default:**
- Works on 3 out of 4 GPU vendors without user configuration
- Users with AMD/Intel GPUs get GPU acceleration immediately
- NVIDIA users can opt-in to CUDA for +10-20% performance
- Minimal dependencies (Vulkan loader typically included in GPU drivers)

---

## Core Components

### 1. Adapter Trait Implementation (from dspy-rs)

**CRITICAL**: This module **implements** the `Adapter` trait provided by dspy-rs, not define its own.

**Source**: https://github.com/krypticmouse/DSRs/blob/main/crates/dspy-rs/src/adapter/mod.rs

```rust
// The Adapter trait is provided by dspy-rs - you IMPLEMENT this trait, not define it
use dspy_rs::adapter::Adapter;
use dspy_rs::{Chat, Example, LM, Message, MetaSignature, Prediction, LmUsage};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use anyhow::Result;
use rig_core::tool::ToolDyn;

// This is what the Adapter trait looks like (from dspy-rs source code):
#[async_trait]
pub trait Adapter: Send + Sync + 'static {
    /// Convert a signature and inputs into a Chat (sequence of Messages)
    fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat;

    /// Parse the model's response message into output fields
    fn parse_response(
        &self,
        signature: &dyn MetaSignature,
        response: Message,
    ) -> HashMap<String, Value>;

    /// Main entry point - orchestrates formatting, inference, and parsing
    async fn call(
        &self,
        lm: Arc<LM>,
        signature: &dyn MetaSignature,
        inputs: Example,
        tools: Vec<Arc<dyn ToolDyn>>,
    ) -> Result<Prediction>;
}
```

### 2. LoadedModel Interface (from Model Pool)

**IMPORTANT**: LlamaCppAdapter does NOT load models. It receives already-loaded models from Model Pool.

```rust
// src/adapters/llamacpp/types.rs

use std::sync::{Arc, Mutex};
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::context::LlamaContext;

/// LoadedModel - Contains a fully loaded llama.cpp model ready for inference
///
/// Provided by Model Pool, similar to CandleAdapter's LoadedModel pattern.
/// The model and context are wrapped in Mutex for thread-safe mutable access.
#[derive(Clone)]
pub struct LoadedModel {
    /// The llama.cpp model (wrapped in Mutex for thread-safe mutable access)
    pub model: Arc<Mutex<LlamaModel>>,

    /// Model context for inference (manages KV cache, sampling state)
    pub context: Arc<Mutex<LlamaContext<'static>>>,

    /// Model name for logging and identification
    pub model_name: String,

    /// Number of GPU layers loaded (for diagnostics)
    pub n_gpu_layers: u32,
}

impl LoadedModel {
    /// Get the model name
    pub fn name(&self) -> &str {
        &self.model_name
    }

    /// Get number of GPU layers
    pub fn gpu_layers(&self) -> u32 {
        self.n_gpu_layers
    }
}
```

### 3. LlamaCppAdapter Implementation

**Location:** `src/adapters/llamacpp/adapter.rs`

```rust
// src/adapters/llamacpp/adapter.rs

use std::sync::Arc;
use std::collections::HashMap;
use async_trait::async_trait;
use dspy_rs::adapter::Adapter;
use dspy_rs::{Chat, Example, LM, LmUsage, Message, MetaSignature, Prediction};
use rig_core::tool::ToolDyn;
use serde_json::Value;

use crate::adapters::llamacpp::{LoadedModel, LlamaCppConfig, LlamaCppError};

/// LlamaCppAdapter - Implements dspy-rs Adapter trait for llama.cpp inference
///
/// Mirrors the CandleAdapter pattern for consistency and dspy-rs compatibility.
/// Key architectural decisions:
/// - Adapter implements dspy-rs Adapter trait directly
/// - Uses Arc<Mutex<LlamaModel>> for thread-safe model access
/// - Uses spawn_blocking for CPU/GPU-bound inference
/// - No separate backend threads (simpler, aligns with dspy-rs stateless pattern)
#[derive(Clone)]
pub struct LlamaCppAdapter {
    /// Loaded model from Model Pool
    model: Arc<LoadedModel>,

    /// Configuration (temperature, max_tokens, etc.)
    config: LlamaCppConfig,
}

impl LlamaCppAdapter {
    /// Create adapter from Model Pool's LoadedModel
    ///
    /// This is the ONLY constructor - no direct model loading.
    /// Model Pool handles all model loading and device setup.
    pub fn from_loaded_model(loaded: Arc<LoadedModel>, config: LlamaCppConfig) -> Self {
        Self {
            model: loaded,
            config,
        }
    }

    /// Get a reference to the configuration
    pub fn config(&self) -> &LlamaCppConfig {
        &self.config
    }

    /// Get a reference to the loaded model
    pub fn model(&self) -> &LoadedModel {
        &self.model
    }

    /// Convert Chat to prompt string
    ///
    /// Uses the same format as CandleAdapter for consistency.
    /// llama.cpp handles chat templating internally if the model supports it.
    fn chat_to_prompt(&self, chat: &Chat) -> String {
        chat.messages
            .iter()
            .map(|msg| match msg {
                Message::System { content } => format!("System: {}", content),
                Message::User { content } => format!("User: {}", content),
                Message::Assistant { content } => format!("Assistant: {}", content),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Generate text using llama.cpp
    ///
    /// Uses spawn_blocking for CPU/GPU-bound work (same pattern as CandleAdapter).
    /// Returns (response_text, prompt_tokens, completion_tokens).
    async fn generate(&self, prompt: &str) -> Result<(String, u64, u64), LlamaCppError> {
        let model = Arc::clone(&self.model);
        let prompt = prompt.to_string();
        let max_tokens = self.config.max_tokens;
        let temperature = self.config.temperature;
        let top_p = self.config.top_p;
        let top_k = self.config.top_k;
        let repeat_penalty = self.config.repeat_penalty;

        // Use spawn_blocking for CPU/GPU-bound work (same pattern as CandleAdapter)
        let result = tokio::task::spawn_blocking(move || {
            // Lock the context for inference
            let mut context = model.context.lock()
                .map_err(|e| LlamaCppError::InferenceFailed(format!("Context lock error: {}", e)))?;

            // Tokenize the prompt
            let tokens = context.tokenize(&prompt, true)
                .map_err(|e| LlamaCppError::TokenizationFailed(e.to_string()))?;

            let prompt_tokens = tokens.len() as u64;

            // Set up sampling parameters
            let mut sampler = llama_cpp_2::sampling::LlamaSampler::new()
                .with_temperature(temperature)
                .with_top_p(top_p)
                .with_top_k(top_k.unwrap_or(40) as i32)
                .with_repeat_penalty(repeat_penalty);

            // Generate tokens
            let mut output_tokens = Vec::new();
            let mut generated = 0;

            // Feed prompt tokens
            context.eval(&tokens, 0)
                .map_err(|e| LlamaCppError::InferenceFailed(format!("Prompt eval error: {}", e)))?;

            // Generate completion
            while generated < max_tokens {
                let token = sampler.sample(&context)
                    .map_err(|e| LlamaCppError::InferenceFailed(format!("Sampling error: {}", e)))?;

                // Check for end of sequence
                if context.is_eog_token(token) {
                    break;
                }

                output_tokens.push(token);
                generated += 1;

                // Feed the token back
                context.eval(&[token], tokens.len() + generated - 1)
                    .map_err(|e| LlamaCppError::InferenceFailed(format!("Token eval error: {}", e)))?;
            }

            let completion_tokens = output_tokens.len() as u64;

            // Decode output tokens to string
            let response = context.detokenize(&output_tokens)
                .map_err(|e| LlamaCppError::InferenceFailed(format!("Detokenization error: {}", e)))?;

            Ok::<_, LlamaCppError>((response, prompt_tokens, completion_tokens))
        })
        .await
        .map_err(|e| LlamaCppError::InferenceFailed(format!("Task join error: {}", e)))??;

        Ok(result)
    }

    /// Generate with retry logic
    async fn generate_with_retry(&self, prompt: &str) -> Result<(String, u64, u64), LlamaCppError> {
        let mut attempt = 0;
        let mut backoff_ms = self.config.initial_backoff_ms;

        loop {
            match self.generate(prompt).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    attempt += 1;
                    if attempt >= self.config.max_retries {
                        return Err(e);
                    }
                    tracing::warn!(
                        "Generation attempt {} failed: {}. Retrying in {}ms",
                        attempt, e, backoff_ms
                    );
                    tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
                    backoff_ms = (backoff_ms * 2).min(self.config.max_backoff_ms);
                }
            }
        }
    }

    /// Format demonstrations into the chat (same as CandleAdapter)
    fn format_demonstrations(&self, signature: &dyn MetaSignature) -> Vec<Message> {
        let mut messages = Vec::new();
        let demos = signature.demos();

        for demo in demos {
            // Format demo inputs as user message
            let mut demo_input = String::new();
            for (field_name, field_value) in demo.data.iter() {
                if demo.input_keys.contains(field_name) {
                    demo_input.push_str(&format!("{}: {}\n", field_name, field_value));
                }
            }
            if !demo_input.is_empty() {
                messages.push(Message::user(demo_input.trim()));
            }

            // Format demo outputs as assistant message
            let mut demo_output = String::new();
            for (field_name, field_value) in demo.data.iter() {
                if demo.output_keys.contains(field_name) {
                    demo_output.push_str(&format!("{}: {}\n", field_name, field_value));
                }
            }
            if !demo_output.is_empty() {
                messages.push(Message::assistant(demo_output.trim()));
            }
        }

        messages
    }

    /// Parse response using field markers (Strategy 1)
    fn parse_with_field_markers(
        &self,
        content: &str,
        output_fields: &[String],
    ) -> Option<HashMap<String, Value>> {
        let mut outputs = HashMap::new();

        for field in output_fields {
            let marker = format!("{}: ", field);
            if let Some(start) = content.find(&marker) {
                let value_start = start + marker.len();
                // Find end (next field marker or end of string)
                let value_end = output_fields
                    .iter()
                    .filter(|f| *f != field)
                    .filter_map(|f| content[value_start..].find(&format!("{}: ", f)))
                    .min()
                    .map(|pos| value_start + pos)
                    .unwrap_or(content.len());

                let value = content[value_start..value_end].trim();
                outputs.insert(field.clone(), Value::String(value.to_string()));
            }
        }

        if outputs.len() == output_fields.len() {
            Some(outputs)
        } else {
            None
        }
    }

    /// Parse response as JSON (Strategy 2)
    fn parse_as_json(
        &self,
        content: &str,
        output_fields: &[String],
    ) -> Option<HashMap<String, Value>> {
        // Try to find JSON in the response
        let json_start = content.find('{');
        let json_end = content.rfind('}');

        if let (Some(start), Some(end)) = (json_start, json_end) {
            if let Ok(json) = serde_json::from_str::<HashMap<String, Value>>(&content[start..=end]) {
                // Check if all output fields are present
                if output_fields.iter().all(|f| json.contains_key(f)) {
                    return Some(json);
                }
            }
        }

        None
    }
}

#[async_trait]
impl Adapter for LlamaCppAdapter {
    /// Convert signature and inputs into Chat
    ///
    /// Same implementation as CandleAdapter for consistency.
    fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat {
        let mut messages = Vec::new();

        // Add system message from signature instruction
        let instruction = signature.instruction();
        if !instruction.is_empty() {
            messages.push(Message::system(instruction));
        }

        // Add demonstrations (few-shot examples) if present
        messages.extend(self.format_demonstrations(signature));

        // Format actual input fields into user message
        let mut user_content = String::new();
        for (field_name, field_value) in inputs.data.iter() {
            if inputs.input_keys.contains(field_name) {
                user_content.push_str(&format!("{}: {}\n", field_name, field_value));
            }
        }

        if !user_content.is_empty() {
            messages.push(Message::user(user_content.trim()));
        }

        Chat::new(messages)
    }

    /// Parse response into output fields
    ///
    /// Uses 3-strategy approach (same as CandleAdapter):
    /// 1. Field marker parsing ("FieldName: value")
    /// 2. JSON parsing fallback
    /// 3. Single-field fallback (use entire response)
    fn parse_response(
        &self,
        signature: &dyn MetaSignature,
        response: Message,
    ) -> HashMap<String, Value> {
        let content = match &response {
            Message::Assistant { content } => content.as_str(),
            _ => return HashMap::new(),
        };

        // Get output field names from signature
        let output_fields: Vec<String> = signature
            .output_fields()
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        if output_fields.is_empty() {
            return HashMap::new();
        }

        // Strategy 1: Try field marker parsing
        if let Some(outputs) = self.parse_with_field_markers(content, &output_fields) {
            return outputs;
        }

        // Strategy 2: Try JSON parsing
        if let Some(outputs) = self.parse_as_json(content, &output_fields) {
            return outputs;
        }

        // Strategy 3: Single-field fallback
        if output_fields.len() == 1 {
            let mut outputs = HashMap::new();
            outputs.insert(
                output_fields[0].clone(),
                Value::String(content.trim().to_string()),
            );
            return outputs;
        }

        // No parsing strategy worked
        HashMap::new()
    }

    /// Main entry point - orchestrates formatting, inference, and parsing
    async fn call(
        &self,
        _lm: Arc<LM>,
        signature: &dyn MetaSignature,
        inputs: Example,
        _tools: Vec<Arc<dyn ToolDyn>>,
    ) -> anyhow::Result<Prediction> {
        // 1. Format inputs into Chat
        let chat = self.format(signature, inputs);

        // 2. Convert Chat to prompt string
        let prompt = self.chat_to_prompt(&chat);

        // 3. Run llama.cpp inference (with retry logic)
        let (response_text, prompt_tokens, completion_tokens) = self
            .generate_with_retry(&prompt)
            .await
            .map_err(|e| anyhow::anyhow!("Generation failed: {}", e))?;

        // 4. Parse response into structured output
        let response_msg = Message::assistant(response_text);
        let outputs = self.parse_response(signature, response_msg);

        // 5. Return Prediction with usage stats
        Ok(Prediction {
            data: outputs,
            lm_usage: LmUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        })
    }
}
```

### 4. Error Types

```rust
// src/adapters/llamacpp/error.rs
//
// Mirrors CandleAdapterError for consistency

use thiserror::Error;

/// Errors that can occur during LlamaCpp adapter operations
#[derive(Debug, Error)]
pub enum LlamaCppError {
    /// Inference operation failed
    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    /// Tokenization failed
    #[error("Tokenization failed: {0}")]
    TokenizationFailed(String),

    /// Context exceeds maximum length
    #[error("Context too long: {actual} tokens > {max} max")]
    ContextTooLong { actual: usize, max: usize },

    /// Token budget exhausted (matches CandleAdapterError)
    #[error("Token budget exhausted: {used}/{limit}")]
    TokenBudgetExhausted { used: usize, limit: usize },

    /// Rate limit exceeded (matches CandleAdapterError)
    #[error("Rate limit exceeded: too many requests")]
    RateLimitExceeded,

    /// Model not loaded
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    /// Backend error (llama.cpp specific)
    #[error("Backend error: {0}")]
    BackendError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Generic error from anyhow
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Result type for LlamaCpp adapter operations
pub type Result<T> = std::result::Result<T, LlamaCppError>;
```

### 5. Thread Safety Design (Phase 3 Implementation)

**Critical Insight:** `LlamaContext` is `!Send + !Sync` in llama-cpp-2

| Type | Send | Sync | Implication |
|------|------|------|-------------|
| `LlamaBackend` | ✅ | ✅ | Can be shared |
| `LlamaModel` | ✅ | ✅ | Can be shared across threads |
| `LlamaContext` | ❌ | ❌ | Cannot cross thread boundaries |

**Solution:** Create context per-request inside `spawn_blocking`

```rust
// LoadedModel only holds model + backend (both Send + Sync)
pub struct LoadedModel {
    pub backend: LlamaBackend,   // Must stay alive
    pub model: LlamaModel,       // Send + Sync
    pub n_ctx: u32,              // Context size
    // NO context stored here - created per-request
}

// In generate_blocking() inside spawn_blocking:
fn generate_blocking(model: &LoadedModel, ...) -> Result<...> {
    // Context created fresh for this request (never crosses threads)
    let mut ctx = model.create_context()?;
    // ... tokenize, decode, sample, detokenize ...
}
```

**Trade-offs:**
- Context creation per-request adds ~1-5ms overhead
- Simpler than thread-local or lazy initialization
- No mutex contention on context
- Inference time (~100-1000ms) dominates

### 6. Configuration

```rust
// src/adapters/llamacpp/config.rs
//
// Mirrors CandleConfig for consistency, with llama.cpp-specific additions

use serde::{Deserialize, Serialize};

/// Configuration for LlamaCppAdapter
///
/// Controls inference parameters and behavior. Mirrors CandleConfig structure
/// for easy migration, with llama.cpp-specific additions (repeat_penalty).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppConfig {
    /// Model identifier (used for logging and debugging)
    pub model_name: String,

    /// Maximum tokens to generate
    pub max_tokens: usize,

    /// Sampling temperature (0.0 - 2.0)
    /// Lower values make output more deterministic
    pub temperature: f32,

    /// Top-p nucleus sampling
    /// Probability mass to consider for sampling
    pub top_p: f32,

    /// Top-k sampling
    /// If Some(k), only consider top k tokens
    pub top_k: Option<usize>,

    /// Repeat penalty (llama.cpp specific, 1.0 = disabled)
    pub repeat_penalty: f32,

    /// Context window size
    /// Maximum number of tokens the model can process
    pub context_length: usize,

    // Production features (matching CandleConfig)
    /// Maximum total tokens allowed across all requests
    pub token_budget_limit: Option<usize>,

    /// Rate limiting: requests per minute
    pub requests_per_minute: Option<u32>,

    /// Number of retry attempts on transient failures
    pub max_retries: u32,

    /// Initial backoff delay in milliseconds
    pub initial_backoff_ms: u64,

    /// Maximum backoff delay in milliseconds
    pub max_backoff_ms: u64,

    /// Enable response caching
    pub enable_cache: bool,

    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,

    /// Enable token-by-token streaming output
    pub enable_streaming: bool,
}

impl Default for LlamaCppConfig {
    /// Default configuration for Qwen2.5-0.5B model
    fn default() -> Self {
        Self {
            model_name: "llama-qwen2.5-0.5b".to_string(),
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: None,  // Disabled by default (same as CandleConfig)
            repeat_penalty: 1.1,  // llama.cpp specific
            context_length: 32768, // Qwen2.5-0.5B supports 32K context
            token_budget_limit: None,
            requests_per_minute: None,
            max_retries: 3,
            initial_backoff_ms: 100,
            max_backoff_ms: 5000,
            enable_cache: false,
            cache_ttl_secs: 300,
            enable_streaming: false,
        }
    }
}

impl LlamaCppConfig {
    /// Create a new configuration with custom model name
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            ..Default::default()
        }
    }

    /// Set maximum tokens to generate
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set sampling temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-p sampling
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set top-k sampling
    pub fn with_top_k(mut self, top_k: Option<usize>) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set repeat penalty (llama.cpp specific)
    pub fn with_repeat_penalty(mut self, repeat_penalty: f32) -> Self {
        self.repeat_penalty = repeat_penalty;
        self
    }

    /// Set context window size
    pub fn with_context_length(mut self, context_length: usize) -> Self {
        self.context_length = context_length;
        self
    }

    /// Enable retry logic with custom attempts
    pub fn with_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Enable response caching
    pub fn with_cache(mut self, enable: bool, ttl_secs: u64) -> Self {
        self.enable_cache = enable;
        self.cache_ttl_secs = ttl_secs;
        self
    }

    /// Enable streaming output
    pub fn with_streaming(mut self, enable: bool) -> Self {
        self.enable_streaming = enable;
        self
    }
}
```

### 6. Module Structure

```rust
// src/adapters/llamacpp/mod.rs

mod adapter;
mod config;
mod error;
mod types;

pub use adapter::LlamaCppAdapter;
pub use config::LlamaCppConfig;
pub use error::LlamaCppError;
pub use types::LoadedModel;
```

---

## Error Handling

### Error Flow

```
LlamaCppError (internal)
    ↓ converts to
anyhow::Error (Adapter trait)
    ↓ handled by
LM struct
    ↓ returns to
DSPy Modules
```

### Error Recovery Strategy

1. **Inference failures**: Retry with exponential backoff (configurable)
2. **Tokenization errors**: Return immediately (not recoverable)
3. **Context overflow**: Return error with details
4. **Backend errors**: Log and return error (may indicate hardware issues)

---

## Configuration

Configuration is set at adapter initialization:

```rust
// Default configuration
let adapter = LlamaCppAdapter::from_loaded_model(loaded_model, LlamaCppConfig::default());

// Custom configuration
let config = LlamaCppConfig::default()
    .with_max_tokens(1024)
    .with_temperature(0.8)
    .with_top_k(Some(50));

let adapter = LlamaCppAdapter::from_loaded_model(loaded_model, config);
```

---

## Testing Strategy

### Port Tests from CandleAdapter

All CandleAdapter tests should be ported to LlamaCppAdapter with minimal changes:

1. **Unit Tests**: Test format(), parse_response() individually
2. **Integration Tests**: Test full call() flow with real GGUF model
3. **Edge Case Tests**: Port all 18 edge case tests
4. **Streaming Tests**: If streaming is implemented

### Test Categories

```rust
// tests/llamacpp_adapter_tests.rs

#[tokio::test]
async fn test_adapter_format() {
    // Test format() produces correct Chat structure
}

#[tokio::test]
async fn test_adapter_parse_response_field_markers() {
    // Test Strategy 1: Field marker parsing
}

#[tokio::test]
async fn test_adapter_parse_response_json() {
    // Test Strategy 2: JSON parsing
}

#[tokio::test]
async fn test_adapter_parse_response_single_field() {
    // Test Strategy 3: Single-field fallback
}

#[tokio::test]
async fn test_adapter_call_integration() {
    // Full integration test with real GGUF model
}

#[tokio::test]
async fn test_dspy_integration() {
    // Test with dspy-rs configure() and Predict
}
```

---

## Performance Considerations

### Optimization Targets
- **Latency**: < 5ms overhead for adapter operations
- **Memory**: < 1MB per adapter instance (model memory handled by Model Pool)
- **Throughput**: Match llama.cpp capacity

### Optimizations Built into llama.cpp
1. **KV Cache**: Automatic (managed by LlamaContext)
2. **Quantization**: GGUF models are pre-quantized (Q4_K_M recommended)
3. **GPU Offloading**: Configurable via n_gpu_layers
4. **Flash Attention**: Available in recent llama.cpp versions

### Expected Performance vs CandleAdapter

| Metric | CandleAdapter | LlamaCppAdapter (expected) |
|--------|---------------|---------------------------|
| Throughput | 4.89 tok/s | 10-30 tok/s (with KV cache) |
| Memory | ~1GB (FP16) | ~300MB (Q4_K_M) |
| Cold start | <2s | <2s |
| GPU Support | NVIDIA, Apple | NVIDIA, AMD, Intel, Apple |

---

## Dependencies

```toml
# Cargo.toml additions

[dependencies]
# llama.cpp Rust bindings
llama-cpp-2 = "0.1"

# Existing dependencies remain
dspy-rs = "0.7.3"
async-trait = "0.1"
tokio = { version = "1", features = ["full"] }
thiserror = "1.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
tracing = "0.1"

[features]
default = ["vulkan"]  # Vulkan as default (broadest GPU support)

cuda = ["llama-cpp-2/cuda"]      # NVIDIA optimization (+10-20% vs Vulkan)
vulkan = ["llama-cpp-2/vulkan"]  # AMD, NVIDIA, Intel GPUs
metal = ["llama-cpp-2/metal"]    # Apple Silicon
cpu = []                          # Fallback (always available)
```

---

## API Examples

### Example 1: Simple Q&A with LlamaCppAdapter

```rust
use ml_crate_dsrs::adapters::llamacpp::{LlamaCppAdapter, LlamaCppConfig};
use ml_crate_dsrs::model_pool::ModelPool;
use dspy_rs::{configure, LM, Predict, Signature, example};

#[derive(Signature)]
struct QuestionAnswer {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Model Pool loads the GGUF model
    let mut model_pool = ModelPool::new("./models".into());
    let loaded_model = model_pool.load_model("qwen2.5-0.5b-instruct-q4_k_m").await?;

    // 2. Create LlamaCppAdapter with loaded model
    let adapter = LlamaCppAdapter::from_loaded_model(loaded_model, LlamaCppConfig::default());

    // 3. Create LM configuration
    let lm = LM::builder()
        .model("llama-qwen2.5-0.5b")
        .temperature(0.7)
        .max_tokens(512)
        .build()
        .await?;

    // 4. Configure dspy-rs with adapter
    configure(lm, adapter);

    // 5. Create predictor
    let qa = Predict::new(QuestionAnswer::new());

    // 6. Execute inference
    let input = example! {
        "question": "input" => "What is the capital of France?"
    };

    let result = qa.forward(input).await?;
    let answer = result.get("answer", None);
    println!("Answer: {}", answer);

    Ok(())
}
```

### Example 2: Chain of Thought Reasoning

```rust
use dspy_rs::{configure, ChainOfThought, Signature, example};

#[derive(Signature)]
struct MathReasoning {
    #[input]
    question: String,

    #[output]
    reasoning: String,

    #[output]
    answer: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Configure with LlamaCpp adapter (same pattern)
    let mut model_pool = ModelPool::new("./models".into());
    let loaded_model = model_pool.load_model("qwen2.5-0.5b-instruct-q4_k_m").await?;
    let adapter = LlamaCppAdapter::from_loaded_model(loaded_model, LlamaCppConfig::default());
    let lm = LM::builder().model("llama-qwen2.5-0.5b").build().await?;
    configure(lm, adapter);

    // Use ChainOfThought module for reasoning
    let cot = ChainOfThought::new(MathReasoning::new());

    let input = example! {
        "question": "input" => "What is 15 * 23?"
    };

    let result = cot.forward(input).await?;
    println!("Reasoning: {}", result.get("reasoning", None));
    println!("Answer: {}", result.get("answer", None));

    Ok(())
}
```

### Example 3: Backend Selection

```rust
// Build with different backends:

// AMD/Intel/NVIDIA (default - Vulkan)
// cargo build

// NVIDIA optimized
// cargo build --features cuda

// macOS
// cargo build --features metal

// CPU only
// cargo build --features cpu
```

---

## Success Criteria

### Minimum Viable Product (MVP)
- [ ] Implements dspy-rs `Adapter` trait correctly
- [ ] Uses dspy-rs types (`Message`, `Value`, `Example`, `Prediction`)
- [ ] Works with `LM` struct for configuration
- [ ] Direct llama.cpp integration works
- [ ] Model Pool integration works
- [ ] Simple text generation works
- [ ] Conversation history formatting works
- [ ] Error handling with retries
- [ ] Integration tests pass
- [ ] Works with DSPy modules (Predict, ChainOfThought)

### Quality Gates
- [ ] All unit tests pass
- [ ] Integration tests with real GGUF model pass
- [ ] No panics or unwraps in production code
- [ ] Clean `cargo clippy` output (0 warnings)
- [ ] Documentation with examples
- [ ] Performance documented

### Backend Verification
- [ ] Vulkan backend works on Windows/Linux
- [ ] CUDA backend works on NVIDIA
- [ ] Metal backend works on macOS
- [ ] CPU fallback works everywhere

---

## Migration from CandleAdapter

### Code Changes Required

| Component | CandleAdapter | LlamaCppAdapter |
|-----------|---------------|-----------------|
| Model type | `candle_transformers::qwen2::Model` | `llama_cpp_2::LlamaModel` |
| Tokenizer | `tokenizers::Tokenizer` | Built into llama.cpp |
| Device | `candle_core::Device` | GPU layers config |
| Model format | Safetensors | GGUF |
| Dependencies | candle-* crates | llama-cpp-2 |

### What Stays the Same

- `Adapter` trait implementation structure
- `format()` method (identical)
- `parse_response()` method (identical)
- `call()` method structure
- `spawn_blocking` pattern for inference
- Error handling patterns
- Configuration patterns
- Test structure

### Migration Steps

1. **Phase 1**: Add llama-cpp-2 dependency alongside Candle
2. **Phase 2**: Implement LlamaCppAdapter mirroring CandleAdapter
3. **Phase 3**: Port all tests
4. **Phase 4**: Validate with real GGUF models
5. **Phase 5**: Remove Candle dependencies (after validation)

---

## Future Enhancements

### Phase 1 Features (Core)
- [ ] Basic generate() implementation
- [ ] generate_stream() for streaming output (matches CandleAdapter)

### Phase 2+ Features
- [ ] Grammar-constrained generation (GBNF support - llama.cpp native)
- [ ] Parallel context processing
- [ ] Speculative decoding support

### Performance Optimizations
- [ ] Context reuse between calls (KV cache management)
- [ ] Batch inference support
- [ ] Prompt caching

---

## Design Decisions

### 1. Mirror CandleAdapter Pattern
- **Rationale**: Minimizes migration effort and ensures compatibility
- **Pattern**: Same Adapter trait implementation, same method signatures
- **Benefit**: Drop-in replacement for CandleAdapter

### 2. spawn_blocking for Inference
- **Rationale**: llama.cpp inference is CPU/GPU-bound, not async
- **Pattern**: Same as CandleAdapter
- **Benefit**: Non-blocking in async runtime

### 3. Arc<Mutex<Model>> Pattern
- **Rationale**: Thread-safe model access (same as CandleAdapter)
- **Pattern**: Standard Rust concurrency pattern
- **Benefit**: Works with Tokio's multi-threaded runtime

### 4. Built-in Tokenization
- **Rationale**: llama.cpp handles tokenization internally
- **Benefit**: No need for separate tokenizer crate, simpler dependency tree

### 5. Vulkan Default
- **Rationale**: Broadest GPU support out of the box
- **Pattern**: Ollama-inspired strategy
- **Benefit**: Works on AMD, Intel, NVIDIA without user configuration

---

## References

### External Resources
- **llama.cpp GitHub**: https://github.com/ggerganov/llama.cpp
- **llama-cpp-2 Rust bindings**: https://github.com/utilityai/llama-cpp-rs
- **GGUF format specification**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **Qwen GGUF models**: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF
- **dspy-rs GitHub**: https://github.com/krypticmouse/DSRs

### Internal Documentation
- **Architecture**: [ARCHv2.md](ARCHv2.md)
- **Migration Strategy**: [03-multi-backend-strategy.md](03-multi-backend-strategy.md)
- **Original Candle Adapter**: [01-candle-adapter.md](01-candle-adapter.md)
- **DSPy Engine**: [02-dspy-engine.md](02-dspy-engine.md)

---

## Changelog

### v1.0.0 (2025-11-26) - MIGRATION COMPLETE 🎉
- ✅ All 6 phases complete
- ✅ Candle dependencies removed (candle-core, candle-nn, candle-transformers, tokenizers)
- ✅ DSPyEngine migrated to LlamaCppAdapter
- ✅ ModelPool migrated to llama.cpp GGUF loading
- ✅ 172 library tests pass, 0 clippy warnings
- ✅ End-to-end tested with real GGUF model on Vulkan GPU
- llama.cpp is now the sole inference backend

### v0.1.1 (2025-11-26) - VALIDATION COMPLETE
- ✅ Validated against dspy-rs v0.7.3 via dspy-researcher
- ✅ Cross-validated against CandleAdapter codebase
- Added missing error variants (TokenBudgetExhausted, RateLimitExceeded)
- Added missing config fields (token_budget_limit, caching, streaming)
- Added generate_stream() to Phase 1 requirements
- Added Validation Report section

### v0.1.0 (2025-11-26) - SPECIFICATION
- Initial specification document
- Architecture aligned with CandleAdapter
- Backend support documented (Vulkan/CUDA/Metal/CPU)
- Migration plan defined
