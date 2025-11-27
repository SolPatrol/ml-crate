# Phase 5: Quality Gates Checklist

## Overview

Ensure the LlamaCpp adapter meets production quality standards before moving to Phase 6 (Model Pool Integration).

**Status:** 🔄 IN PROGRESS

---

## Completed Items ✅

| Task | Status | Notes |
|------|--------|-------|
| All unit tests pass | ✅ | 23 tests (14 unit + 9 integration) |
| Integration tests pass | ✅ | All 9 pass with shared OnceLock model |
| Clippy clean | ✅ | 0 warnings |
| Performance benchmarking | ✅ | ~5.6s GPU vs ~16s CPU (3x speedup) |

---

## Remaining Items

| Task | Status | Notes |
|------|--------|-------|
| Audit for panics/unwraps | ⏳ | Review production code paths |
| Documentation with examples | ⏳ | README, inline docs, usage examples |

---

## Task 1: Panic/Unwrap Audit

**Goal:** Ensure no panics or unwraps in production code paths. Tests can use `.expect()`.

### Files to Audit

| File | Status | Notes |
|------|--------|-------|
| `src/adapters/llamacpp/adapter.rs` | ⏳ | Main adapter logic |
| `src/adapters/llamacpp/types.rs` | ⏳ | LoadedModel |
| `src/adapters/llamacpp/config.rs` | ⏳ | Configuration |
| `src/adapters/llamacpp/error.rs` | ⏳ | Error types |

### What to Look For

1. **`.unwrap()`** - Replace with `?` or proper error handling
2. **`.expect()`** - OK in tests, replace in production code
3. **`panic!()`** - Should never be in production paths
4. **`unreachable!()`** - Verify these are truly unreachable
5. **Array indexing `[i]`** - Use `.get(i)` with proper error handling

### Acceptable Patterns

```rust
// OK in tests
let model = get_shared_model().expect("Model not available");

// OK for truly invariant conditions
let n = NonZeroU32::new(self.n_ctx); // n_ctx validated at construction

// AVOID in production
let value = map["key"]; // Use map.get("key").ok_or(Error)?
```

---

## Task 2: Documentation

### Inline Documentation

| Item | Status | Notes |
|------|--------|-------|
| Module-level docs (`//!`) | ⏳ | Each file needs overview |
| Public struct docs | ⏳ | `LlamaCppAdapter`, `LoadedModel`, `LlamaCppConfig` |
| Public method docs | ⏳ | All pub fn need `///` docs |
| Error variant docs | ⏳ | Document when each error occurs |

### Usage Examples

| Example | Status | Notes |
|---------|--------|-------|
| Basic inference | ⏳ | Load model, run single query |
| dspy-rs integration | ⏳ | `configure()` with Predict |
| Custom config | ⏳ | Temperature, max_tokens, etc. |

### Example: Basic Inference

```rust
// examples/llamacpp_basic.rs
use ml_crate_dsrs::adapters::llamacpp::{LlamaCppAdapter, LlamaCppConfig, LoadedModel};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load model with GPU acceleration
    let loaded = LoadedModel::load(
        "models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        99,   // GPU layers (99 = all)
        2048, // Context size
    )?;

    // Create adapter
    let adapter = LlamaCppAdapter::from_loaded_model(
        Arc::new(loaded),
        LlamaCppConfig::default()
            .with_max_tokens(256)
            .with_temperature(0.7),
    );

    // Run inference (using dspy-rs)
    // ... see dspy-rs integration example

    Ok(())
}
```

### Example: dspy-rs Integration

```rust
// examples/llamacpp_dspy.rs
use ml_crate_dsrs::adapters::llamacpp::{LlamaCppAdapter, LlamaCppConfig, LoadedModel};
use dspy_rs::{configure, LM, Predict, Signature, example};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load model
    let loaded = LoadedModel::load(
        "models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        99, 2048,
    )?;

    let adapter = LlamaCppAdapter::from_loaded_model(
        Arc::new(loaded),
        LlamaCppConfig::default(),
    );

    // Configure dspy-rs
    let lm = LM::builder()
        .model("local-llamacpp".to_string())
        .base_url("http://localhost:0".to_string())
        .build()
        .await?;

    configure(lm, adapter);

    // Use with Predict
    #[derive(Signature)]
    struct QA {
        #[input]
        question: String,
        #[output]
        answer: String,
    }

    let predictor = Predict::new(QA::new());
    let result = predictor.forward(example! {
        "question": "input" => "What is 2+2?"
    }).await?;

    println!("Answer: {}", result.get("answer", None));
    Ok(())
}
```

---

## Verification Commands

```bash
# Run all tests
cargo test --features vulkan adapters::llamacpp
cargo test --features vulkan adapters::llamacpp -- --ignored --test-threads=1

# Check for panics/unwraps in production code (manual grep)
grep -rn "\.unwrap()" src/adapters/llamacpp/ --include="*.rs" | grep -v "#\[cfg(test)\]" | grep -v "mod tests"
grep -rn "\.expect(" src/adapters/llamacpp/ --include="*.rs" | grep -v "#\[cfg(test)\]" | grep -v "mod tests"
grep -rn "panic!" src/adapters/llamacpp/ --include="*.rs" | grep -v "#\[cfg(test)\]"

# Clippy with extra lints
cargo clippy --features vulkan -- -W clippy::unwrap_used -W clippy::expect_used

# Build docs
cargo doc --features vulkan --no-deps --open
```

---

## Success Criteria

- [ ] No `.unwrap()` in production code paths
- [ ] No `.expect()` in production code paths (tests OK)
- [ ] No `panic!()` in production code
- [ ] All public items have `///` documentation
- [ ] At least 2 usage examples in `examples/`
- [ ] `cargo doc` generates without warnings
- [ ] All tests still pass after changes

---

## Changelog

### v0.1.0 (2025-11-26) - INITIAL CHECKLIST
- Created Phase 5 checklist
- Documented audit requirements
- Added example templates
- Defined success criteria
