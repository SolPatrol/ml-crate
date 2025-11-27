# Phase 4: Testing & Validation Checklist

## Overview

Complete the remaining testing and validation tasks for the LlamaCpp adapter to ensure full dspy-rs compatibility.

**Status:** ✅ COMPLETE

---

## Completed Items ✅

| Task | Status | Notes |
|------|--------|-------|
| Unit tests for LlamaCppAdapter | ✅ | 23 tests (14 unit + 9 integration) |
| Integration tests with real GGUF model | ✅ | All 9 pass with shared model |
| Verify token counts are accurate | ✅ | Prompt + completion tokens tracked |
| Vulkan on Windows | ✅ | NVIDIA GTX 1070, 25/25 layers offloaded |
| CPU fallback | ✅ | Verified with `--features cpu` |
| Edge case unit tests | ✅ | 6 tests for empty/whitespace/unicode/special chars |
| dspy-rs configure() test | ✅ | `test_dspy_configure()` added |
| Clippy clean | ✅ | 0 warnings |
| Shared model via OnceLock | ✅ | Mirrors production pattern |

---

## Task 1: Edge Case Unit Tests ✅ COMPLETE

**File:** `src/adapters/llamacpp/adapter.rs` (tests module)

| Test | Status | Description |
|------|--------|-------------|
| `test_parse_response_empty` | ✅ | Empty response handling |
| `test_parse_response_whitespace_only` | ✅ | Whitespace-only response |
| `test_parse_response_special_chars` | ✅ | Quotes, ampersands, angle brackets |
| `test_parse_response_unicode` | ✅ | Chinese, emoji, Cyrillic |
| `test_format_empty_instruction` | ✅ | No system message when empty |
| `test_format_very_long_input` | ✅ | 5000+ char input handling |

**Note:** These tests are **unit tests** that don't require a real model.

---

## Task 2: dspy-rs `configure()` Test ✅ COMPLETE

**File:** `src/adapters/llamacpp/adapter.rs` (tests module)

| Step | Status | Notes |
|------|--------|-------|
| Create test that calls `dspy_rs::configure(lm, adapter)` | ✅ | `test_dspy_configure()` |
| Verify adapter is set as global singleton | ✅ | Verified via configure() |

**Test:** `test_dspy_configure` - marked `#[ignore = "requires GGUF model"]`

---

## Task 3: Test with DSPy Predict Module ⏸️ OPTIONAL

**Status:** Deferred - tests dspy-rs code, not our adapter

The `test_call_integration` test already verifies the full pipeline:
- format() → generate() → parse_response()

Predict module testing would test dspy-rs internals, not our adapter.

---

## Task 4: Test with DSPy ChainOfThought Module ⏸️ OPTIONAL

**Status:** Deferred - tests dspy-rs code, not our adapter

Similar to Predict - CoT tests would verify dspy-rs, not our adapter.

---

## Task 5: Full dspy-rs v0.7.3 Compatibility ✅ VERIFIED

| Step | Status | Notes |
|------|--------|-------|
| `Adapter` trait implementation | ✅ | All 3 methods implemented |
| `format()` works | ✅ | With signatures & demos |
| `parse_response()` all 3 strategies | ✅ | Field markers, JSON, single-field |
| `call()` end-to-end | ✅ | `test_call_integration` passes |
| `MetaSignature` compatibility | ✅ | MockSignature implements it |
| Few-shot demonstrations | ✅ | `test_demonstrations_formatting` |

---

## Task 6: Backend Verification

| Backend | Platform | Status | Notes |
|---------|----------|--------|-------|
| Vulkan | Windows | ✅ | NVIDIA GTX 1070, 25/25 layers |
| CUDA | Windows | ⏸️ | Deferred - needs CUDA toolkit |
| Metal | macOS | ⏸️ | Deferred - needs macOS |
| CPU | All | ✅ | Verified |

---

## Known Limitations

### LlamaBackend Singleton
The `LlamaBackend::init()` can only be called once per process.

**Solution:** Tests use a shared model via `OnceLock` in `adapter.rs`:
```rust
static SHARED_MODEL: OnceLock<Arc<LoadedModel>> = OnceLock::new();
```
This mirrors production where one model instance serves all requests.

---

## Verification Commands

```bash
# Run all unit tests (fast, no model required)
cargo test --features vulkan adapters::llamacpp
# Result: 14 passed, 9 ignored

# Run integration tests with real model (all pass with shared model)
cargo test --features vulkan adapters::llamacpp -- --ignored --test-threads=1
# Result: 9 passed

# Run clippy
cargo clippy --features vulkan
# Result: 0 warnings
```

---

## Test Summary

| Category | Count | Status |
|----------|-------|--------|
| Unit tests (no model) | 14 | ✅ All pass |
| Integration tests | 9 | ✅ All pass (shared model) |
| Total | 23 | ✅ |

---

## Success Criteria

- [x] All 14 unit tests pass
- [x] All 9 integration tests pass (shared model)
- [x] `dspy_rs::configure()` test added
- [x] Edge case tests added (6 new tests)
- [x] All 3 parse strategies verified
- [x] Few-shot demonstrations work correctly
- [x] Clippy clean (0 warnings)
- [x] No regressions from CandleAdapter behavior
- [x] Shared model via OnceLock (mirrors production)

---

## Changelog

### v1.1.0 (2025-11-26) - SHARED MODEL FIX
- Fixed integration tests to use shared model via `OnceLock`
- All 9 integration tests now pass together
- Removed redundant types.rs tests (covered by adapter.rs)
- Pattern mirrors production: one model instance for all requests

### v1.0.0 (2025-11-26) - PHASE 4 COMPLETE
- Added 6 edge case unit tests (no model required)
- Added `test_dspy_configure()` integration test
- All 14 unit tests pass
- Integration tests verified with Vulkan GPU
- Clippy clean (0 warnings)

### v0.1.0 (2025-11-26) - INITIAL CHECKLIST
- Created Phase 4 checklist
- Documented remaining tasks
- Added test code templates
- Defined success criteria
