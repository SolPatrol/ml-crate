# Multi-Backend Strategy: llama.cpp Migration

**Version**: 1.2.0
**Last Updated**: 2025-11-26

---

## Summary

Migrate from Candle to **llama.cpp** as the unified inference backend to support all GPU hardware (NVIDIA, AMD, Apple, Intel) with a single codebase and model format.

---

## Current State

| Component | Technology | Hardware Support |
|-----------|------------|------------------|
| Inference Engine | Candle | NVIDIA (CUDA), Apple (Metal) |
| Model Format | Safetensors | - |
| Missing | - | ❌ AMD GPUs |

---

## Proposed State

| Component | Technology | Hardware Support |
|-----------|------------|------------------|
| Inference Engine | llama.cpp (via Rust bindings) | All |
| Model Format | GGUF | - |
| Coverage | - | ✅ NVIDIA, AMD, Apple, Intel, CPU |

---

## Why llama.cpp?

### Hardware Coverage

```
llama.cpp
    │
    ├── CUDA    → NVIDIA GPUs
    ├── Vulkan  → AMD GPUs, Intel GPUs
    ├── Metal   → Apple Silicon
    └── CPU     → Fallback (AVX2/AVX512)
```

### Comparison with Alternatives

| Option | AMD Support | Effort | Model Format |
|--------|-------------|--------|--------------|
| **llama.cpp** | ✅ Vulkan | Low | GGUF |
| Burn + ONNX | ❌ Missing ops (SiLU, RMSNorm) | Blocked | ONNX |
| Candle | ❌ No Vulkan/wgpu | - | Safetensors |
| Contribute to Burn | ✅ Eventually | High | ONNX |

### Why Not Burn?

Burn's `burn-wgpu` backend supports all hardware via WebGPU/Vulkan, but `burn-import` (ONNX converter) is missing critical operators used by modern LLMs:

- ❌ **SiLU** (activation function)
- ❌ **RMSNorm** (normalization)

All modern LLMs (Qwen, Llama, Mistral, Phi) use SiLU + RMSNorm. Until Burn adds these operators, ONNX import is not viable for our models.

---

## Architecture

### Before (Candle)

```
Safetensors Model
       │
       ▼
┌─────────────────┐
│ candle-core     │
│ candle-transform│
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
  CUDA      Metal
 NVIDIA     Apple
```

### After (llama.cpp)

```
GGUF Model (quantized)
       │
       ▼
┌─────────────────┐
│  llama-cpp-rs   │
│  (Rust bindings)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   llama.cpp     │
│  (C++ engine)   │
└────────┬────────┘
         │
    ┌────┼────┬────────┬────────┐
    ▼    ▼    ▼        ▼        ▼
  CUDA Vulkan Metal   CPU    OpenCL
 NVIDIA  AMD  Apple  Any     Legacy
        Intel
```

---

## Implementation Plan

### Phase 1: Add llama.cpp Dependency

```toml
# Cargo.toml
[dependencies]
llama-cpp-2 = "0.1"

[features]
default = ["vulkan"]  # Vulkan as default (broadest GPU support)

cuda = ["llama-cpp-2/cuda"]      # NVIDIA optimization (+10-20% vs Vulkan)
vulkan = ["llama-cpp-2/vulkan"]  # AMD, NVIDIA, Intel GPUs
metal = ["llama-cpp-2/metal"]    # Apple Silicon (auto-enabled on macOS ARM64)
cpu = []                          # Fallback (always available)
```

### Backend Selection Strategy (Ollama-inspired)

**Default to Vulkan** for broadest out-of-the-box support:

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

**User experience:**
```bash
# AMD/Intel/NVIDIA user - just works
cargo build

# NVIDIA user wanting max performance
cargo build --features cuda

# macOS user
cargo build --features metal

# No GPU / CI environment
cargo build --features cpu
```

### Phase 2: Create LlamaCppAdapter (mirrors CandleAdapter pattern)

**IMPORTANT**: The `LlamaCppAdapter` must implement the dspy-rs `Adapter` trait directly, following the same pattern as the existing `CandleAdapter`. This ensures compatibility with dspy-rs's singleton LM pattern via `configure()`.

```rust
// src/adapters/llamacpp/adapter.rs
//
// This mirrors the CandleAdapter pattern from specs/01-candle-adapter.md
// Key architectural decisions:
// - Adapter implements dspy-rs Adapter trait directly
// - Uses Arc<Mutex<LlamaModel>> for thread-safe model access
// - Uses spawn_blocking for CPU/GPU-bound inference (same as Candle)
// - No separate backend threads (simpler, aligns with dspy-rs stateless pattern)

use std::sync::{Arc, Mutex};
use async_trait::async_trait;
use dspy_rs::adapter::Adapter;
use dspy_rs::{Chat, Example, LM, LmUsage, Message, MetaSignature, Prediction};
use rig_core::tool::ToolDyn;
use serde_json::Value;
use std::collections::HashMap;

/// LoadedModel - Contains a fully loaded llama.cpp model ready for inference
///
/// Provided by Model Pool, similar to CandleAdapter's LoadedModel
#[derive(Clone)]
pub struct LoadedModel {
    /// The llama.cpp model (wrapped in Mutex for thread-safe mutable access)
    pub model: Arc<Mutex<llama_cpp_2::LlamaModel>>,

    /// Model context for inference
    pub context: Arc<Mutex<llama_cpp_2::LlamaContext>>,

    /// Model name for logging
    pub model_name: String,
}

/// LlamaCppAdapter - Implements dspy-rs Adapter trait for llama.cpp inference
///
/// Mirrors the CandleAdapter pattern for consistency and dspy-rs compatibility.
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

    /// Convert Chat to prompt string (same as CandleAdapter)
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

    /// Generate text using llama.cpp (uses spawn_blocking like CandleAdapter)
    async fn generate(&self, prompt: &str) -> Result<(String, u64, u64), LlamaCppError> {
        let model = Arc::clone(&self.model);
        let prompt = prompt.to_string();
        let max_tokens = self.config.max_tokens;
        let temperature = self.config.temperature;

        // Use spawn_blocking for CPU/GPU-bound work (same pattern as CandleAdapter)
        let result = tokio::task::spawn_blocking(move || {
            let mut context = model.context.lock()
                .map_err(|e| LlamaCppError::InferenceFailed(format!("Lock error: {}", e)))?;

            // llama.cpp inference
            let response = context.generate(&prompt, max_tokens, temperature)?;

            // Get token counts from llama.cpp
            let prompt_tokens = context.prompt_token_count() as u64;
            let completion_tokens = context.completion_token_count() as u64;

            Ok::<_, LlamaCppError>((response, prompt_tokens, completion_tokens))
        })
        .await
        .map_err(|e| LlamaCppError::InferenceFailed(format!("Task join error: {}", e)))??;

        Ok(result)
    }
}

#[async_trait]
impl Adapter for LlamaCppAdapter {
    /// Convert signature and inputs into Chat (same as CandleAdapter)
    fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat {
        let mut messages = Vec::new();

        // Add system message from signature instruction
        let instruction = signature.instruction();
        if !instruction.is_empty() {
            messages.push(Message::system(instruction));
        }

        // Add demonstrations (few-shot examples) if present
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

    /// Parse response into output fields (same 3-strategy approach as CandleAdapter)
    fn parse_response(
        &self,
        signature: &dyn MetaSignature,
        response: Message,
    ) -> HashMap<String, Value> {
        // Same implementation as CandleAdapter - 3 parsing strategies:
        // 1. Field marker parsing ("FieldName: value")
        // 2. JSON parsing fallback
        // 3. Single-field fallback (use entire response)

        // ... (see CandleAdapter implementation in 01-candle-adapter.md)
        HashMap::new() // Placeholder - copy from CandleAdapter
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

        // 3. Run llama.cpp inference (with real token counts)
        let (response_text, prompt_tokens, completion_tokens) = self.generate(&prompt).await
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

### Phase 3: Update Model Pool

Model Pool provides `LoadedModel` to the adapter (same pattern as CandleAdapter):

```rust
// src/model_pool/mod.rs

use std::sync::{Arc, Mutex};
use std::path::PathBuf;

pub struct ModelPool {
    /// Currently loaded model
    loaded_model: Option<Arc<LoadedModel>>,

    /// Model directory path
    model_dir: PathBuf,
}

impl ModelPool {
    pub fn new(model_dir: PathBuf) -> Self {
        Self {
            loaded_model: None,
            model_dir,
        }
    }

    /// Load a GGUF model and return LoadedModel for adapter
    pub async fn load_model(&mut self, model_name: &str) -> Result<Arc<LoadedModel>> {
        let model_path = self.model_dir.join(format!("{}.gguf", model_name));

        // Load model with llama.cpp
        let params = llama_cpp_2::LlamaModelParams::default()
            .with_n_gpu_layers(99);  // Offload all layers to GPU

        let model = llama_cpp_2::LlamaModel::load_from_file(&model_path, params)?;
        let context = model.new_context(llama_cpp_2::LlamaContextParams::default())?;

        let loaded = Arc::new(LoadedModel {
            model: Arc::new(Mutex::new(model)),
            context: Arc::new(Mutex::new(context)),
            model_name: model_name.to_string(),
        });

        self.loaded_model = Some(Arc::clone(&loaded));
        Ok(loaded)
    }

    /// Get currently loaded model (for adapter creation)
    pub fn get_loaded_model(&self) -> Option<Arc<LoadedModel>> {
        self.loaded_model.clone()
    }
}
```

### Phase 4: Integration with dspy-rs (same as CandleAdapter)

```rust
// Usage example - mirrors CandleAdapter integration

use dspy_rs::{configure, LM, Predict, Signature, example};
use ml_crate_dsrs::model_pool::ModelPool;
use ml_crate_dsrs::adapters::llamacpp::{LlamaCppAdapter, LlamaCppConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Model Pool loads the GGUF model
    let mut model_pool = ModelPool::new("./models".into());
    let loaded_model = model_pool.load_model("qwen2.5-0.5b-instruct-q4_k_m").await?;

    // 2. Create LlamaCppAdapter with loaded model (same pattern as CandleAdapter)
    let adapter = LlamaCppAdapter::from_loaded_model(loaded_model, LlamaCppConfig::default());

    // 3. Configure dspy-rs with adapter (same as CandleAdapter)
    let lm = LM::builder()
        .model("llama-qwen2.5-0.5b")
        .temperature(0.7)
        .max_tokens(512)
        .build()
        .await?;

    configure(lm, adapter);

    // 4. Use DSPy predictor (identical to CandleAdapter usage)
    #[derive(Signature)]
    struct QA {
        #[input]
        question: String,
        #[output]
        answer: String,
    }

    let predictor = Predict::new(QA::new());
    let inputs = example! {
        "question": "input" => "What is 2+2?"
    };

    let result = predictor.forward(inputs).await?;
    println!("Answer: {}", result.get("answer", None));

    Ok(())
}
```

---

## Model Format Migration

### Current: Safetensors
- Location: `models/qwen2.5-0.5b/`
- Size: ~1GB (FP16)

### New: GGUF
- Location: `models/qwen2.5-0.5b-q4_k_m.gguf`
- Size: ~300MB (Q4_K_M quantized)

### Obtaining GGUF Models

```bash
# Option 1: Download pre-quantized from HuggingFace
# https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF

# Option 2: Convert from HuggingFace format
python llama.cpp/convert_hf_to_gguf.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --outfile qwen2.5-0.5b-instruct-f16.gguf

# Option 3: Quantize existing GGUF
./llama.cpp/llama-quantize \
    qwen2.5-0.5b-instruct-f16.gguf \
    qwen2.5-0.5b-instruct-q4_k_m.gguf \
    Q4_K_M
```

---

## Rust Binding: `llama-cpp-2`

**Selected**: [`llama-cpp-2`](https://github.com/utilityai/llama-cpp-rs) (utilityai/llama-cpp-rs)

| Attribute | Value |
|-----------|-------|
| Package | `llama-cpp-2` |
| Stars | 405 ⭐ |
| Commits | 1,366 |
| Last Release | Nov 2025 |
| Backends | CUDA, Vulkan, Metal |

**Why this crate:**
- Most actively maintained
- Stays current with llama.cpp upstream
- Has all required backends: CUDA, Vulkan, Metal
- Low-level API gives full control

---

## Migration Checklist

> **Detailed Specification**: See [04-llamacpp-adapter.md](04-llamacpp-adapter.md) for full implementation details.

### Phase 0: Specification ✅ COMPLETE
- [x] Create LlamaCpp adapter specification ([04-llamacpp-adapter.md](04-llamacpp-adapter.md))
- [x] Validate architecture against dspy-rs v0.7.3
- [x] Cross-validate against CandleAdapter codebase
- [x] Update ARCHv2.md with llama.cpp architecture ([ARCHv2.md](ARCHv2.md))

### Phase 1: Dependencies & Core Types ✅ COMPLETE
- [x] Add `llama-cpp-2` dependency with feature flags (vulkan default)
- [x] Create `src/adapters/llamacpp/` module structure
- [x] Add `LlamaCppError` enum (10 variants, matches CandleAdapterError + BackendError)
- [x] Add `LlamaCppConfig` struct (15 fields, 11 builder methods)
- [x] Implement `LoadedModel` struct (mirrors CandleAdapter)
- [x] Create `LlamaCppAdapter` stub (constructor + getters)
- [x] Verify builds: Vulkan ✅, CUDA ✅, CPU ✅
- [x] Document Windows build setup ([06-windows-build-setup.md](06-windows-build-setup.md))

### Phase 2: Adapter Implementation ✅ COMPLETE
- [x] Implement `LlamaCppAdapter` with dspy-rs `Adapter` trait
- [x] Implement `from_loaded_model()` constructor
- [x] Port `format()` from CandleAdapter
- [x] Port `format_demonstrations()` for few-shot learning
- [x] Port `parse_response()` with 3-strategy parsing:
  - [x] Strategy 1: Field marker parsing
  - [x] Strategy 2: JSON parsing fallback
  - [x] Strategy 3: Single-field fallback
- [x] Implement `chat_to_prompt()` helper
- [x] Implement `generate()` with `spawn_blocking` pattern (placeholder for llama-cpp-2 API)
- [x] Implement `generate_with_retry()` for error recovery
- [ ] Implement `generate_stream()` for streaming output (deferred to Phase 2B)
- [x] Implement full `call()` method (format → generate → parse)

### Phase 3: Real llama-cpp-2 Integration ✅ COMPLETE
- [x] Replace placeholder types with real llama-cpp-2 types (`LlamaBackend`, `LlamaModel`, `LlamaContext`)
- [x] Implement `LoadedModel::load()` for GGUF model loading
- [x] Implement `generate_blocking()` with full inference pipeline
- [x] Thread safety: context created per-request (solves `!Send + !Sync`)
- [x] Add `seed` config parameter for reproducible sampling
- [x] Fix Windows CRT mismatch (esaxx-rs vs llama-cpp-sys-2)
- [x] Verify GPU inference (Vulkan on NVIDIA GTX 1070)

### Phase 4: Testing & Validation ✅ COMPLETE
- [x] Unit tests for LlamaCppAdapter (23 tests: 14 unit + 9 integration)
- [x] Integration tests with real GGUF model (all 9 pass with shared model)
- [x] Verify token counts are accurate (prompt + completion tokens tracked)
- [x] Edge case unit tests (6 tests for empty/whitespace/unicode/special chars)
- [x] Verify dspy-rs `configure()` works correctly (`test_dspy_configure()` added)
- [x] Shared model via OnceLock (mirrors production pattern)
- [x] Clippy clean (0 warnings)
- [x] Backend verification:
  - [x] Vulkan on Windows (NVIDIA GTX 1070, 25/25 layers offloaded)
  - [ ] CUDA on NVIDIA (deferred - requires full recompile)
  - [ ] Metal on macOS (deferred - requires macOS)
  - [x] CPU fallback (verified)

### Phase 5: Quality Gates 🔄 IN PROGRESS
- [x] All unit tests pass (23 tests: 14 unit + 9 integration)
- [x] Integration tests with real GGUF model pass
- [x] Clean `cargo clippy` output (0 warnings)
- [ ] No panics or unwraps in production code (audit pending)
- [ ] Documentation with examples
- [x] Performance benchmarking: ~5.6s GPU vs ~16s CPU (3x speedup)
- See [10-phase5-checklist.md](10-phase5-checklist.md) for detailed checklist

### Phase 6: Model Pool Integration & Cleanup
- [x] Download Qwen GGUF model (qwen2.5-0.5b-instruct-q4_k_m.gguf)
- [ ] Update `ModelPool` to load GGUF models
- [ ] Return `Arc<LoadedModel>` for adapter creation
- [ ] Configure GPU layers based on Hardware Manager
- [ ] Update all documentation
- [ ] Remove Candle dependencies (after full validation)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| FFI complexity | Use well-maintained bindings (llama-cpp-2) |
| Build issues (CMake, C++) | Document build requirements, CI testing |
| llama.cpp breaking changes | Pin to stable version, test upgrades |

---

## Future Considerations

- **Burn**: Monitor `burn-import` for SiLU + RMSNorm support. If added, reconsider ONNX path for pure-Rust solution.
- **llama.cpp**: Actively developed; stay current with releases for performance improvements and new model support.

---

## References

### Internal Documentation
- [LlamaCpp Adapter Specification](04-llamacpp-adapter.md) - Full implementation spec
- [ARCHv2](ARCHv2.md) - Updated architecture with llama.cpp
- [CandleAdapter Specification](01-candle-adapter.md) - Reference implementation
- [DSPy Engine](02-dspy-engine.md) - Integration context

### External Resources
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [llama-cpp-2 Rust bindings](https://github.com/utilityai/llama-cpp-rs)
- [GGUF format specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Qwen GGUF models on HuggingFace](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
