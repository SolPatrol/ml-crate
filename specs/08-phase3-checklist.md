# Phase 3: Real llama-cpp-2 Integration Checklist

## Overview

Replace placeholder implementation with actual llama-cpp-2 inference using the downloaded GGUF model.

**Model Path:** `models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf`

**Status:** ✅ COMPLETE (2025-11-26)

---

## Key llama-cpp-2 Types (from research)

| Type | Purpose | Thread Safety |
|------|---------|---------------|
| `LlamaBackend` | Initialize llama.cpp backend | Once at startup |
| `LlamaModel` | Loaded GGUF model | `Send + Sync` (shareable) |
| `LlamaContext` | Inference state & KV cache | `!Send + !Sync` (thread-local) |
| `LlamaBatch` | Token batch for processing | Per-inference |
| `LlamaSampler` | Token sampling chain | Per-inference |

---

## Task 1: Update types.rs with Real Types ✅ COMPLETE

**File:** `src/adapters/llamacpp/types.rs`

| Step | Description | Status |
|------|-------------|--------|
| 1 | Replace `pub type LlamaModel = ()` with `llama_cpp_2::model::LlamaModel` | [x] |
| 2 | Replace `pub type LlamaContext = ()` with `llama_cpp_2::context::LlamaContext<'_>` | [x] |
| 3 | Update `LoadedModel` struct to hold actual types | [x] |
| 4 | Add `LlamaBackend` holder (needs to stay alive) | [x] |
| 5 | Update `LoadedModel::load()` constructor | [x] |
| 6 | Write unit tests for type construction | [x] |

**Implementation Notes:**
- Re-exported llama-cpp-2 types: `LlamaBackend`, `LlamaModel`, `LlamaContext`, `LlamaModelParams`, `LlamaContextParams`
- `LoadedModel` holds `backend`, `model`, `model_name`, `n_gpu_layers`, `n_ctx`
- Context created per-request via `create_context()` method (solves `!Send + !Sync` issue)

---

## Task 2: Create Model Loading Function ✅ COMPLETE

**File:** `src/adapters/llamacpp/types.rs`

| Step | Description | Status |
|------|-------------|--------|
| 1 | Add `LoadedModel::load(path: &str, n_gpu_layers: u32, n_ctx: u32) -> Result<LoadedModel>` | [x] |
| 2 | Initialize `LlamaBackend::init()` | [x] |
| 3 | Create `LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers)` | [x] |
| 4 | Load model with `LlamaModel::load_from_file(&backend, path, &params)` | [x] |
| 5 | Context creation deferred to `create_context()` (per-request) | [x] |
| 6 | Return `LoadedModel` with model + backend | [x] |
| 7 | Add error handling for file not found, load errors | [x] |

**Implementation Notes:**
- Context NOT stored in LoadedModel (created per-request to handle `!Send + !Sync`)
- `create_context()` method creates fresh context with `n_ctx` parameter

---

## Task 3: Implement generate_blocking() ✅ COMPLETE

**File:** `src/adapters/llamacpp/adapter.rs`

| Step | Description | Status |
|------|-------------|--------|
| 1 | Import llama-cpp-2 types: `LlamaBatch`, `LlamaSampler`, `AddBos` | [x] |
| 2 | Tokenize prompt: `model.str_to_token(prompt, AddBos::Always)` | [x] |
| 3 | Check context length: `if tokens.len() > n_ctx` | [x] |
| 4 | Create sampler chain with config params | [x] |
| 5 | Create batch and add prompt tokens | [x] |
| 6 | Decode initial batch: `ctx.decode(&mut batch)` | [x] |
| 7 | Generation loop: sample, check EOG, decode | [x] |
| 8 | Detokenize output: `model.token_to_str(token)` | [x] |
| 9 | Return `(text, prompt_tokens, completion_tokens)` | [x] |

**Implementation Notes:**
- Uses `LlamaSampler::chain_simple()` with temp, top_p, top_k, penalties, dist
- Seed from config (random if None)
- Proper EOG detection via `model.is_eog_token(token)`
- Token counts tracked accurately

---

## Task 4: Update Error Types ✅ COMPLETE

**File:** `src/adapters/llamacpp/error.rs`

| Step | Description | Status |
|------|-------------|--------|
| 1 | Add `ModelNotLoaded(String)` variant | [x] |
| 2 | Add `BackendError(String)` variant | [x] |
| 3 | Add `TokenizationFailed(String)` variant | [x] |
| 4 | Add `DecodeFailed(String)` variant | [x] |
| 5 | Add `SamplingFailed(String)` variant | [x] |
| 6 | Implement `From` traits for llama-cpp-2 errors | [x] |

**Implementation Notes:**
- All error variants support string messages for llama-cpp-2 error conversion
- `BackendError` covers backend init and context creation failures

---

## Task 5: Add Config Parameters ✅ COMPLETE

**File:** `src/adapters/llamacpp/config.rs`

| Step | Description | Status |
|------|-------------|--------|
| 1 | Add `context_length: usize` (default 2048) | [x] |
| 2 | Add `n_batch: usize` - not needed (handled internally) | [x] N/A |
| 3 | Add `n_threads: Option<usize>` - not needed (auto) | [x] N/A |
| 4 | Add `seed: Option<u64>` (default None = random) | [x] |
| 5 | Update builder methods | [x] |
| 6 | Update tests | [x] |

**Implementation Notes:**
- `context_length` defaults to 2048
- `seed` for reproducible sampling (None = random via `rand::random()`)
- `with_seed()` builder method added

---

## Task 6: Integration Test with Real Model ✅ COMPLETE

**File:** `src/adapters/llamacpp/types.rs` and `src/adapters/llamacpp/adapter.rs`

| Step | Description | Status |
|------|-------------|--------|
| 1 | Load GGUF model from `models/gguf/` | [x] |
| 2 | Create adapter with loaded model | [x] |
| 3 | Test simple generation | [x] |
| 4 | Verify token counts are accurate | [x] |
| 5 | Test with different prompts | [x] |
| 6 | Test error handling (invalid model path) | [x] |

**Tests Created:**
- `test_loaded_model_load` - Model loading with GPU layers
- `test_loaded_model_create_context` - Context creation
- `test_generate_real` - Real inference with token counting

**Verified:**
- GPU inference works (Vulkan backend, NVIDIA GTX 1070)
- 25/25 layers offloaded to GPU
- ~5.6s inference time (vs ~16s on CPU) - 3x speedup
- Token counts accurate

---

## Task 7: Verification ✅ COMPLETE

| Check | Command | Status |
|-------|---------|--------|
| Build (CPU) | `cargo check --no-default-features --features cpu` | [x] |
| Build (Vulkan) | `cargo check --features vulkan` | [x] |
| Tests | `cargo test --test-threads=1 -- --ignored` | [x] |
| Clippy | `cargo clippy` | [x] (0 warnings) |
| Integration | Real model tests pass | [x] |

**Notes:**
- Tests require `--test-threads=1` due to `LlamaBackend::init()` singleton
- Tests with `#[ignore]` require GGUF model file present

---

## Files Modified

1. `src/adapters/llamacpp/types.rs` - Real llama-cpp-2 types ✅
2. `src/adapters/llamacpp/adapter.rs` - `generate_blocking()` implementation ✅
3. `src/adapters/llamacpp/config.rs` - Added `seed` parameter ✅
4. `src/adapters/llamacpp/error.rs` - Error variants already complete ✅
5. `.cargo/config.toml` - Windows CRT fix (`/MD` flags) ✅

---

## Thread Safety Solution

**Problem:** `LlamaContext` is `!Send + !Sync` in llama-cpp-2

**Solution:** Create context per-request inside `spawn_blocking`

```rust
// LoadedModel holds only Send + Sync types
pub struct LoadedModel {
    pub backend: LlamaBackend,  // Send + Sync
    pub model: LlamaModel,      // Send + Sync
    pub n_ctx: u32,
    // NO context stored - created per-request
}

// Context created fresh for each inference
fn generate_blocking(model: &LoadedModel, ...) -> Result<...> {
    let mut ctx = model.create_context()?;  // Thread-local
    // ... inference ...
}
```

**Trade-offs:**
- Context creation adds ~1-5ms overhead per request
- Simpler than mutex/thread-local approaches
- No contention on shared context
- Inference time (~100-1000ms) dominates

---

## Windows Build Fix

**Problem:** CRT mismatch between `esaxx-rs` (MT_StaticRelease) and `llama-cpp-sys-2` (MD_DynamicRelease)

**Solution:** Force `/MD` for all C/C++ builds in `.cargo/config.toml`:

```toml
[env]
CFLAGS = "/MD"
CXXFLAGS = "/MD"
```

---

## Changelog

### v1.0.0 (2025-11-26) - PHASE 3 COMPLETE
- Replaced placeholder types with real llama-cpp-2 types
- Implemented `LoadedModel::load()` for GGUF model loading
- Implemented `generate_blocking()` with full inference pipeline
- Added `seed` config parameter for reproducible sampling
- Fixed Windows CRT mismatch (esaxx-rs vs llama-cpp-sys-2)
- Verified GPU inference (Vulkan) on NVIDIA GTX 1070
- All tests passing, clippy clean (0 warnings)
- Thread safety solved via per-request context creation

### v0.1.0 (2025-11-26) - INITIAL CHECKLIST
- Created Phase 3 checklist
- Documented llama-cpp-2 types and thread safety
- Defined all tasks with verification steps
