# Phase 1: Dependencies & Core Types - Implementation Checklist

**Status**: ✅ COMPLETE
**Completed**: 2025-11-26
**Reference Specs**: [03-multi-backend-strategy.md](03-multi-backend-strategy.md), [04-llamacpp-adapter.md](04-llamacpp-adapter.md)

---

## Prerequisites

- [x] Verify Rust toolchain is up to date (`rustup update`)
- [x] Verify CMake is installed (required for llama.cpp build)
- [x] Verify C++ compiler is available (MSVC on Windows)
- [x] Verify Vulkan SDK is installed (for default backend)
- [x] Windows-specific: See [06-windows-build-setup.md](06-windows-build-setup.md) for full setup

---

## Task 1: Add llama-cpp-2 Dependency ✅

**File**: `Cargo.toml`

### Checklist
- [x] Add `llama-cpp-2 = "0.1"` to `[dependencies]`
- [x] Add feature flags section with vulkan default
- [x] Run `cargo check` to verify dependency resolves
- [x] Run `cargo build --features vulkan` passes
- [x] Run `cargo build --features cuda` passes
- [x] Run `cargo build --features cpu` passes
- [x] Document build issues in [06-windows-build-setup.md](06-windows-build-setup.md)

---

## Task 2: Create Module Structure ✅

**Directory**: `src/adapters/llamacpp/`

### Checklist
- [x] Create `src/adapters/llamacpp/` directory
- [x] Create `src/adapters/llamacpp/mod.rs`
- [x] Create `src/adapters/llamacpp/adapter.rs`
- [x] Create `src/adapters/llamacpp/config.rs`
- [x] Create `src/adapters/llamacpp/error.rs`
- [x] Create `src/adapters/llamacpp/types.rs`
- [x] Update `src/adapters/mod.rs` to include `pub mod llamacpp;`
- [x] Verify module structure with `cargo check`

---

## Task 3: Implement LlamaCppError ✅

**File**: `src/adapters/llamacpp/error.rs`

### 3.1 Error Variants (match CandleAdapterError + BackendError)

| Variant | Description | Status |
|---------|-------------|--------|
| `InferenceFailed(String)` | Inference operation failed | ✅ |
| `TokenizationFailed(String)` | Tokenization failed | ✅ |
| `ContextTooLong { actual, max }` | Context exceeds max length | ✅ |
| `TokenBudgetExhausted { used, limit }` | Token budget exhausted | ✅ |
| `RateLimitExceeded` | Rate limit exceeded | ✅ |
| `ModelNotLoaded(String)` | Model not loaded | ✅ |
| `BackendError(String)` | llama.cpp backend error | ✅ NEW |
| `ConfigError(String)` | Configuration error | ✅ |
| `InvalidParameter(String)` | Invalid parameter | ✅ |
| `Other(anyhow::Error)` | Generic error | ✅ |

### Checklist
- [x] Create `error.rs` with `LlamaCppError` enum
- [x] Add all error variants from table above
- [x] Implement `#[derive(Debug, Error)]` with thiserror
- [x] Add `pub type Result<T> = std::result::Result<T, LlamaCppError>;`
- [x] Add `#[error(transparent)] Other(#[from] anyhow::Error)` for conversion
- [x] Write unit test for error display messages (`test_error_display`)
- [x] Write unit test for error conversion (`test_error_conversion`)
- [x] Export from `mod.rs`

---

## Task 4: Implement LlamaCppConfig ✅

**File**: `src/adapters/llamacpp/config.rs`

### 4.1 Config Fields (15 fields)

| Field | Type | Default | Status |
|-------|------|---------|--------|
| `model_name` | `String` | `"llama-qwen2.5-0.5b"` | ✅ |
| `max_tokens` | `usize` | `512` | ✅ |
| `temperature` | `f32` | `0.7` | ✅ |
| `top_p` | `f32` | `0.9` | ✅ |
| `top_k` | `Option<usize>` | `None` | ✅ |
| `repeat_penalty` | `f32` | `1.1` | ✅ NEW |
| `context_length` | `usize` | `32768` | ✅ |
| `token_budget_limit` | `Option<usize>` | `None` | ✅ |
| `requests_per_minute` | `Option<u32>` | `None` | ✅ |
| `max_retries` | `u32` | `3` | ✅ |
| `initial_backoff_ms` | `u64` | `100` | ✅ |
| `max_backoff_ms` | `u64` | `5000` | ✅ |
| `enable_cache` | `bool` | `false` | ✅ |
| `cache_ttl_secs` | `u64` | `300` | ✅ |
| `enable_streaming` | `bool` | `false` | ✅ |

### 4.2 Builder Methods (11 methods)

| Method | Status |
|--------|--------|
| `new(model_name)` | ✅ |
| `with_max_tokens` | ✅ |
| `with_temperature` | ✅ |
| `with_top_p` | ✅ |
| `with_top_k` | ✅ |
| `with_repeat_penalty` | ✅ |
| `with_context_length` | ✅ |
| `with_retries` | ✅ |
| `with_cache` | ✅ |
| `with_streaming` | ✅ |
| `with_token_budget` | ✅ |
| `with_rate_limit` | ✅ |

### Checklist
- [x] Create `config.rs` with `LlamaCppConfig` struct
- [x] Add `#[derive(Debug, Clone, Serialize, Deserialize)]`
- [x] Implement all fields from table above
- [x] Implement `Default` trait with default values
- [x] Implement all builder methods
- [x] Write unit test for `Default::default()` (`test_default_config`)
- [x] Write unit test for builder pattern (`test_config_builder`)
- [x] Export from `mod.rs`

---

## Task 5: Implement LoadedModel ✅

**File**: `src/adapters/llamacpp/types.rs`

### Checklist
- [x] Create `types.rs` with `LoadedModel` struct
- [x] Add `#[derive(Clone)]` (Arc fields are Clone)
- [x] Implement `new()` constructor
- [x] Implement `name()` method
- [x] Implement `gpu_layers()` method
- [x] Write unit tests (`test_loaded_model_new`, `test_loaded_model_clone`)
- [x] Export from `mod.rs`

---

## Task 6: Create Adapter Stub ✅

**File**: `src/adapters/llamacpp/adapter.rs`

### Checklist
- [x] Create `adapter.rs` with `LlamaCppAdapter` struct
- [x] Add `#[derive(Clone)]`
- [x] Implement `from_loaded_model()` constructor
- [x] Implement `config()` getter
- [x] Implement `model()` getter
- [x] Add TODO comment for Phase 2 Adapter trait implementation
- [x] Write unit tests (`test_adapter_from_loaded_model`, `test_adapter_clone`)
- [x] Export from `mod.rs`

---

## Task 7: Module Exports ✅

**File**: `src/adapters/llamacpp/mod.rs`

### Checklist
- [x] Add `mod adapter;`
- [x] Add `mod config;`
- [x] Add `mod error;`
- [x] Add `mod types;`
- [x] Add `pub use adapter::LlamaCppAdapter;`
- [x] Add `pub use config::LlamaCppConfig;`
- [x] Add `pub use error::{LlamaCppError, Result};`
- [x] Add `pub use types::LoadedModel;`
- [x] Add dspy-rs type re-exports for convenience

---

## Task 8: Update Parent Module ✅

**File**: `src/adapters/mod.rs`

### Checklist
- [x] Add `pub mod llamacpp;` alongside existing `pub mod candle;`
- [x] Add `pub use llamacpp::LlamaCppAdapter;`
- [x] Verify both modules compile with `cargo check`

---

## Task 9: Verification ✅

### Build Verification
- [x] `cargo check` passes
- [x] `cargo check --features vulkan` passes (default)
- [x] `cargo check --features cuda` passes
- [x] `cargo check --features cpu` passes

### Test Verification
- [x] `cargo test` passes (all 178 tests)
- [x] New unit tests pass:
  - [x] `test_error_display`
  - [x] `test_error_conversion`
  - [x] `test_default_config`
  - [x] `test_config_builder`
  - [x] `test_loaded_model_new`
  - [x] `test_loaded_model_clone`
  - [x] `test_adapter_from_loaded_model`
  - [x] `test_adapter_clone`

---

## Completion Summary

Phase 1 is complete:

1. ✅ `llama-cpp-2` dependency added with feature flags (vulkan, cuda, metal, cpu)
2. ✅ `src/adapters/llamacpp/` module structure created (5 files)
3. ✅ `LlamaCppError` enum implemented (10 variants)
4. ✅ `LlamaCppConfig` struct implemented (15 fields, 11 builder methods)
5. ✅ `LoadedModel` struct implemented
6. ✅ `LlamaCppAdapter` stub created (constructor + getters)
7. ✅ All builds pass (vulkan, cuda, cpu)
8. ✅ All tests pass (178 tests)
9. ✅ Windows build setup documented in [06-windows-build-setup.md](06-windows-build-setup.md)

---

## Next Phase

Proceed to **Phase 2: Adapter Implementation**:
- Implement dspy-rs `Adapter` trait
- Port `format()` from CandleAdapter
- Port `parse_response()` with 3-strategy parsing
- Implement `generate()` with `spawn_blocking`
- Implement `call()` method

See [04-llamacpp-adapter.md](04-llamacpp-adapter.md) for Phase 2 details.
