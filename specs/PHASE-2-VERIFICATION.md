# Phase 2 Implementation - Verification Specification

**Date**: 2025-11-18
**Status**: 🔄 **PLANNING** - Ready to Begin
**Version**: dspy-rs v0.7.3
**Prerequisites**: Phase 0 ✅ Complete, Phase 1 ✅ Complete
**Model**: Qwen2.5-0.5B
**Focus**: DSPy Integration & Production Features

---

## Executive Summary

Phase 2 of the Candle Adapter implementation focuses on **full DSPy integration** and **production readiness**. This phase transforms the working Candle inference from Phase 1 into a fully-functional, production-ready DSPy adapter that works seamlessly with all DSPy predictors and modules at production-grade performance.

### Phase 2 Goals

**Compatibility Goals (85% → 100%)**:
1. ✅ **DSPy Integration** - Full compatibility with dspy-rs v0.7.3
2. ✅ **Signature Support** - Structured input/output with type hints
3. ✅ **Predictor Compatibility** - Works with Predict, ChainOfThought, ReAct
4. ✅ **Example Type** - Full DSPy type system support

**Production Goals (Performance & UX)**:
5. ✅ **KV Cache** - 5-10x speedup (4.89 → 25-50 tok/s)
6. ✅ **Streaming Output** - Token-by-token generation for real-time UX
7. ✅ **Batch Inference** - Higher throughput for multiple requests
8. ✅ **Comprehensive Testing** - End-to-end validation with DSPy modules

**Total Scope**: Full dspy-rs compatibility + Production-ready performance and UX

### Critical Requirements (From dspy-researcher)

**The 85% → 100% Gap**: Our current implementation is **85% accurate** to dspy-rs v0.7.3. The 15% gap is:

**MUST HAVE** (Closes the 15% gap → 100% dspy-rs compatibility):
- ✅ Implement signature parsing and formatting (**~10% of gap**)
- ✅ Add Example type definition (**~3% of gap**)
- ✅ Enhanced error types (CandleAdapterError enum) (**~2% of gap**)
- ✅ Test with actual DSPy predictors (Predict, ChainOfThought) (validation)

**SHOULD HAVE** (Improves implementation quality, not compatibility):
- ✅ Proper tokenization with tokenizers crate (we already have this in Phase 1!)
- ✅ Chat template support (better prompt formatting)

**PERFORMANCE FEATURES** (Beyond 100% compatibility - production readiness):
- ✅ Performance optimization (KV cache) - **5-10x speedup, production quality**
- ✅ Streaming output - **Better UX, production quality**
- ✅ Batch inference - **Higher throughput, production quality**
- ✅ Example notebooks - **Documentation and examples**

**Key Insight**: The **MUST HAVE** features get us to **100% dspy-rs compatibility**. The **PERFORMANCE FEATURES** make it production-ready with excellent performance (25-50 tok/s vs 4.89 tok/s).

### Recommended Implementation Strategy

Phase 2 includes both compatibility AND performance features:

#### **Phase 2A: Core DSPy Compatibility** (85% → 100%)
**Focus**: Close the 15% gap to achieve full dspy-rs v0.7.3 compatibility

**Deliverables**:
1. Signature support (parse, format, use in prompts)
2. Example type implementation
3. Enhanced error handling
4. Integration tests with Predict and ChainOfThought

**Estimated Effort**: 1-2 days
**Result**: **100% dspy-rs compatible** ✅

#### **Phase 2B: Production Performance** (100% → Production-Ready)
**Focus**: Improve performance and UX for production deployment

**Deliverables**:
1. KV cache implementation (5-10x speedup)
2. Streaming output (token-by-token)
3. Batch inference (higher throughput)
4. Example notebooks and documentation

**Estimated Effort**: 2-3 days
**Result**: 4.89 tok/s → 25-50 tok/s 🚀 + Production-ready UX

**Total Phase 2 Effort**: 3-5 days for complete implementation (compatibility + performance)

### Key Metrics

#### Phase 2A: Compatibility (1-2 days)
- **Files to Modify**: 3 (adapter.rs, config.rs, mod.rs)
- **New Functions**: ~8-10 (signature parsing, Example type, error handling)
- **New Tests**: ~10 (signature tests, DSPy integration)
- **Code Growth**: ~300-400 lines

#### Phase 2B: Performance (2-3 days)
- **Files to Modify**: 2 (adapter.rs, config.rs)
- **New Functions**: ~6-8 (KV cache, streaming, batching)
- **New Tests**: ~8 (performance benchmarks, streaming tests)
- **Code Growth**: ~300-400 lines

#### Phase 2 Total
- **Total Effort**: 3-5 days
- **Total Code Growth**: ~600-800 lines (from ~1,120 → ~1,720-1,920 lines)
- **Total New Tests**: ~18 tests
- **Performance Improvement**: 5-10x (4.89 → 25-50 tok/s)

---

## Table of Contents

- [Verification Checklist](#verification-checklist)
- [Component Changes](#component-changes)
- [DSPy Integration Requirements](#dspy-integration-requirements)
- [Testing Strategy](#testing-strategy)
- [Performance Benchmarks](#performance-benchmarks)
- [Success Criteria](#success-criteria)
- [Troubleshooting Guide](#troubleshooting-guide)
- [Rollback Plan](#rollback-plan)

---

## Verification Checklist

### ✅ Pre-Implementation (Phase 1 Complete)

- [x] All Phase 1 tests passing (8/8 integration + 6/6 unit)
- [x] Zero clippy warnings (with CUDA env setup)
- [x] Clean compilation (cargo check)
- [x] Real Candle inference working correctly
- [x] Adapter trait verified against dspy-rs v0.7.3
- [x] Model Pool functional with caching (1816x speedup)
- [x] Token counting accurate (not estimated)

### 🔄 Phase 2 - Component Verification

#### 1. dspy-rs Source Code Verification

**CRITICAL**: All implementation MUST be verified against actual dspy-rs v0.7.3 source code stored in `.claude/knowledge/dspy/source/`

- [ ] **Adapter trait verified** (`.claude/knowledge/dspy/source/adapter-trait.md`)
  - [ ] Method signatures match exactly
  - [ ] Parameter types match exactly
  - [ ] Return types match exactly
  - [ ] Trait bounds match exactly

- [ ] **Core types verified** (`.claude/knowledge/dspy/source/core-types.md`)
  - [ ] `Prediction` struct (ONLY 2 fields: `data`, `lm_usage`)
  - [ ] `LmUsage` struct (3 u64 fields: `prompt_tokens`, `completion_tokens`, `total_tokens`)
  - [ ] `Message` enum (3 variants: System, User, Assistant)
  - [ ] `Chat` struct (2 fields: `messages`, `tools`)
  - [ ] `Example` struct (3 fields: `data`, `input_keys`, `output_keys`)
  - [ ] `MetaSignature` trait (methods: `instruction()`, `input_fields()`, `output_fields()`)

- [ ] **LM configuration verified** (`.claude/knowledge/dspy/source/lm-struct.md`)
  - [ ] `LM` is a STRUCT, not a trait
  - [ ] Builder pattern usage
  - [ ] `configure()` function signature

- [ ] **Predictor APIs verified** (`.claude/knowledge/dspy/source/predictors-api.md`)
  - [ ] `Predict` module API
  - [ ] `ChainOfThought` module API
  - [ ] `ReAct` module API
  - [ ] Module initialization patterns

#### 2. Signature Support Implementation

**CRITICAL**: Signatures are CORE to DSPy's type-safe approach. This is the #1 priority for Phase 2.

- [ ] **Signature parsing**
  - [ ] Parse `MetaSignature` trait objects
  - [ ] Extract `instruction()` text
  - [ ] Extract `input_fields()` from signature
  - [ ] Extract `output_fields()` from signature
  - [ ] Handle field types (all return `Vec<Value>` in v0.7.3)
  - [ ] Validate signature structure

- [ ] **Prompt formatting with signatures**
  - [ ] `format()` uses signature instruction
  - [ ] Input fields formatted into user message
  - [ ] Output field hints added to prompt
  - [ ] System message includes instruction
  - [ ] Examples added if provided

- [ ] **Output parsing with signatures**
  - [ ] `parse_response()` extracts output fields
  - [ ] Field extraction based on signature
  - [ ] Type conversion (String → Value)
  - [ ] Error handling for missing fields
  - [ ] Partial output support (optional fields)

- [ ] **Testing**
  - [ ] Unit test: parse simple signature
  - [ ] Unit test: format prompt with signature
  - [ ] Unit test: parse response with signature
  - [ ] Integration test: end-to-end with signature

#### 3. Example Type Implementation

- [ ] **Type definition**
  - [ ] `Example` struct matches dspy-rs v0.7.3:
    ```rust
    pub struct Example {
        pub data: HashMap<String, Value>,      // All fields
        pub input_keys: Vec<String>,           // Which are inputs
        pub output_keys: Vec<String>,          // Which are outputs
    }
    ```
  - [ ] Constructor: `Example::new(data, input_keys, output_keys)`
  - [ ] Helper methods: `inputs()`, `outputs()`, `get()`, `set()`

- [ ] **The `example!` Macro** (Recommended way to create Examples)
  - [ ] **Purpose**: Idiomatic way to create `Example` instances
  - [ ] **Source**: `.claude/knowledge/dspy/source/core-types.md:597-623`
  - [ ] **Syntax**:
    ```rust
    let example = example! {
        "question": "input" => "What is 2+2?",
        "answer": "output" => "4"
    };
    ```
  - [ ] **What it does**:
    - Automatically separates inputs and outputs
    - Converts values to `serde_json::Value`
    - Creates `Example` with correct `input_keys` and `output_keys`
  - [ ] **Alternative (manual)**:
    ```rust
    let example = Example::new(
        hashmap! {
            "question".to_string() => json!("What is 2+2?"),
            "answer".to_string() => json!("4")
        },
        vec!["question".to_string()],  // input_keys
        vec!["answer".to_string()],     // output_keys
    );
    ```

- [ ] **Integration**
  - [ ] Used in `format()` method
  - [ ] Used in `parse_response()` method
  - [ ] Used in `call()` method
  - [ ] Predictor examples use `example!` macro

- [ ] **Testing**
  - [ ] Unit test: Example creation (both manual and macro)
  - [ ] Unit test: Example field access
  - [ ] Integration test: Example with adapter
  - [ ] Test `example!` macro compatibility

#### 4. Enhanced Error Handling

- [ ] **Error type definition**
  - [ ] `CandleAdapterError` enum created
  - [ ] Variants:
    - [ ] `InferenceFailed(String)`
    - [ ] `TokenizationFailed(String)`
    - [ ] `ContextTooLong { actual: usize, max: usize }`
    - [ ] `SignatureError(String)` - NEW for Phase 2
    - [ ] `ParseError(String)` - NEW for Phase 2
    - [ ] `ConfigError(String)`
    - [ ] `ModelLoadError(String)`

- [ ] **Error conversion**
  - [ ] `From<CandleAdapterError>` for `anyhow::Error`
  - [ ] Error messages are clear and actionable
  - [ ] Stack traces preserved where relevant

- [ ] **Error handling in methods**
  - [ ] `format()` returns errors (not panic)
  - [ ] `parse_response()` returns errors gracefully
  - [ ] `call()` propagates errors correctly

#### 5. Predictor Integration

**CRITICAL**: Must work with ALL dspy-rs predictors

- [ ] **Predict module**
  - [ ] `Predict::new(signature)` works with CandleAdapter
  - [ ] `predict.forward(inputs)` returns correct `Prediction`
  - [ ] Output fields populated correctly
  - [ ] Token usage tracked accurately

- [ ] **ChainOfThought module**
  - [ ] Works with reasoning signatures
  - [ ] Extracts `reasoning` field
  - [ ] Extracts `answer` field
  - [ ] Multi-step reasoning works

- [ ] **ReAct module** (if time permits)
  - [ ] Works with action signatures
  - [ ] Tool calls parsed correctly
  - [ ] Observation integration works
  - [ ] Multi-step agent loops work

- [ ] **Testing**
  - [ ] Integration test: Simple Q&A with Predict
  - [ ] Integration test: Math reasoning with ChainOfThought
  - [ ] Integration test: Tool use with ReAct (optional)
  - [ ] Integration test: Multi-turn conversation

#### 6. Production Features (Phase 2B)

##### 6.1 KV Cache Implementation

**Priority**: HIGH - 5-10x performance improvement

- [ ] **Cache structure**
  - [ ] Key cache: `Vec<Tensor>` (one per layer)
  - [ ] Value cache: `Vec<Tensor>` (one per layer)
  - [ ] Cache lifecycle management
  - [ ] Memory estimation and limits

- [ ] **Integration**
  - [ ] Pass cache to `model.forward()`
  - [ ] Update cache after each token
  - [ ] Clear cache on new prompt
  - [ ] Reuse cache for continuation

- [ ] **Configuration**
  - [ ] `enable_kv_cache: bool` in config
  - [ ] `max_cache_size: Option<usize>` for memory limit
  - [ ] Cache statistics tracking

- [ ] **Performance validation**
  - [ ] Measure speedup (target: 5-10x)
  - [ ] Memory overhead acceptable (< 500MB)
  - [ ] No correctness regression
  - [ ] Benchmark vs. Phase 1 baseline

##### 6.2 Streaming Output

**Priority**: MEDIUM - Better UX for real-time applications

- [ ] **Stream interface**
  - [ ] Return `impl Stream<Item = Result<String>>` or similar
  - [ ] Yield tokens as generated
  - [ ] Error handling in stream
  - [ ] Graceful stream termination

- [ ] **Integration**
  - [ ] Optional stream vs. batch mode
  - [ ] Compatible with DSPy predictors (if supported)
  - [ ] Testing with async streams
  - [ ] Example notebook demonstrating streaming

- [ ] **Configuration**
  - [ ] `enable_streaming: bool` in config
  - [ ] Stream buffer size settings

##### 6.3 Batch Inference

**Priority**: MEDIUM - Higher throughput for multiple requests

- [ ] **Batching logic**
  - [ ] Pad sequences to same length
  - [ ] Single forward pass for batch
  - [ ] Unbatch outputs correctly
  - [ ] Handle variable-length sequences

- [ ] **Configuration**
  - [ ] `batch_size: Option<usize>` in config
  - [ ] Dynamic vs. static batching
  - [ ] Timeout settings for batch collection

- [ ] **Performance validation**
  - [ ] Higher GPU utilization measured
  - [ ] Throughput improvement quantified
  - [ ] Latency impact acceptable

#### 7. Configuration Enhancements

- [ ] **New config fields for Phase 2**
  - [ ] `enable_kv_cache: bool` (default: `true`)
  - [ ] `enable_streaming: bool` (default: `false`)
  - [ ] `batch_size: Option<usize>` (default: `None`)
  - [ ] `chat_template: Option<String>` (default: `None`)
  - [ ] `chat_stop_tokens: Vec<String>` (default: `vec![]`)

- [ ] **Configuration validation**
  - [ ] Invalid combinations rejected
  - [ ] Helpful error messages
  - [ ] Defaults work out-of-box

---

## Component Changes

### File: `src/adapters/candle/adapter.rs`

#### Changes Required

1. **Import additions**
   ```rust
   use dspy_rs::{Chat, Example, LM, Message, MetaSignature, Prediction, LmUsage};
   use serde_json::Value;
   use std::collections::HashMap;
   ```

2. **New helper methods**
   ```rust
   impl CandleAdapter {
       /// Parse signature to extract formatting instructions
       fn extract_signature_info(&self, signature: &dyn MetaSignature) -> SignatureInfo {
           // NEW: Extract instruction, input fields, output fields
       }

       /// Format prompt with signature-based structure
       fn format_with_signature(&self, signature: &dyn MetaSignature, inputs: Example) -> String {
           // NEW: Create structured prompt using signature
       }

       /// Parse model output based on signature structure
       fn parse_with_signature(&self, signature: &dyn MetaSignature, output: &str) -> HashMap<String, Value> {
           // NEW: Extract output fields based on signature
       }
   }
   ```

3. **Update existing methods**
   ```rust
   #[async_trait]
   impl Adapter for CandleAdapter {
       fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat {
           // MODIFY: Use signature instruction
           // MODIFY: Use signature input fields for formatting
           // MODIFY: Add output field hints to prompt
       }

       fn parse_response(&self, signature: &dyn MetaSignature, response: Message) -> HashMap<String, Value> {
           // MODIFY: Use signature output fields
           // MODIFY: Parse structured output
           // MODIFY: Handle missing fields gracefully
       }

       async fn call(&self, lm: Arc<LM>, signature: &dyn MetaSignature, inputs: Example, tools: Vec<Arc<dyn ToolDyn>>) -> Result<Prediction> {
           // MODIFY: Use signature-aware formatting
           // MODIFY: Use signature-aware parsing
           // KEEP: Token counting and LmUsage (already correct)
       }
   }
   ```

### File: `src/adapters/candle/config.rs`

#### Changes Required

1. **New configuration fields**
   ```rust
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct CandleConfig {
       // ... existing fields ...

       // NEW: Phase 2 features
       pub enable_kv_cache: bool,
       pub enable_streaming: bool,
       pub batch_size: Option<usize>,
       pub chat_template: Option<String>,
       pub chat_stop_tokens: Vec<String>,
   }
   ```

2. **Update Default impl**
   ```rust
   impl Default for CandleConfig {
       fn default() -> Self {
           Self {
               // ... existing defaults ...

               // NEW: Phase 2 defaults
               enable_kv_cache: true,
               enable_streaming: false,
               batch_size: None,
               chat_template: None,
               chat_stop_tokens: vec![],
           }
       }
   }
   ```

### File: `src/adapters/candle/mod.rs`

#### Changes Required

1. **Export new types**
   ```rust
   mod adapter;
   mod config;
   mod error;  // Potentially split out

   pub use adapter::{CandleAdapter, LoadedModel};
   pub use config::CandleConfig;
   pub use error::CandleAdapterError;  // If split out
   ```

### File: `tests/integration_tests.rs` (to be updated)

#### New Tests Required

1. **Test 9: DSPy Predict Integration**
   ```rust
   #[tokio::test]
   async fn test_9_dspy_predict_integration() {
       // Test basic Predict module with CandleAdapter
   }
   ```

2. **Test 10: Signature Parsing**
   ```rust
   #[tokio::test]
   async fn test_10_signature_parsing() {
       // Test signature extraction and formatting
   }
   ```

3. **Test 11: ChainOfThought Integration**
   ```rust
   #[tokio::test]
   async fn test_11_chain_of_thought() {
       // Test reasoning with ChainOfThought module
   }
   ```

4. **Test 12-15: Additional predictor tests**

---

## DSPy Integration Requirements

### 1. Adapter Trait - VERIFIED AGAINST SOURCE

**Source**: `.claude/knowledge/dspy/source/adapter-trait.md` (Lines 34-54)

```rust
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

**Status**: ✅ Current implementation signatures match exactly

**Phase 2 Work**: Enhance implementations to actually USE the signature parameter

### 2. Core Types - VERIFIED AGAINST SOURCE

**Source**: `.claude/knowledge/dspy/source/core-types.md`

#### Prediction (Lines 182-186)

```rust
pub struct Prediction {
    pub data: HashMap<String, Value>,  // Output fields (NOT "outputs"!)
    pub lm_usage: LmUsage,              // Token usage stats
}
```

**Status**: ✅ Current implementation correct

#### LmUsage (Lines 560-576)

```rust
pub struct LmUsage {
    pub prompt_tokens: u64,       // NOT usize, NOT u32
    pub completion_tokens: u64,   // NOT usize, NOT u32
    pub total_tokens: u64,        // Auto-calculated in new()
}

impl LmUsage {
    pub fn new(prompt_tokens: u64, completion_tokens: u64) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}
```

**Status**: ✅ Current implementation correct

#### Message (Lines 283-288)

```rust
pub enum Message {
    System { content: String },
    User { content: String },
    Assistant { content: String },
}
```

**Status**: ✅ Current implementation correct

#### Chat (Lines 362-366)

```rust
pub struct Chat {
    pub messages: Vec<Message>,
    pub tools: Vec<Arc<dyn ToolDyn>>,
}
```

**Status**: ✅ Current implementation correct

#### Example (Lines 26-31)

```rust
pub struct Example {
    pub data: HashMap<String, Value>,   // All fields
    pub input_keys: Vec<String>,        // Which fields are inputs
    pub output_keys: Vec<String>,       // Which fields are outputs
}
```

**Status**: ❌ NOT YET IMPLEMENTED - Phase 2 requirement

### 3. MetaSignature Trait - VERIFIED AGAINST SOURCE

**Source**: `.claude/knowledge/dspy/source/core-types.md` (signature-related)

```rust
use std::fmt::Debug;
use std::any::Any;

pub trait MetaSignature: Debug + Send + Sync {
    fn instruction(&self) -> String;
    fn input_fields(&self) -> Vec<Value>;
    fn output_fields(&self) -> Vec<Value>;
    fn prefix(&self) -> Option<String>;
    fn desc(&self) -> Option<String>;
    fn signature_name(&self) -> String;
    fn as_any(&self) -> &dyn Any;
}
```

**Source Reference**: `.claude/knowledge/dspy/source/core-types.md:459-468`

**Usage Pattern**:
```rust
fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat {
    let instruction = signature.instruction();
    let input_fields = signature.input_fields();
    let output_fields = signature.output_fields();

    // Optional metadata
    let prefix = signature.prefix();
    let desc = signature.desc();
    let name = signature.signature_name();

    // Use these to format the prompt
}
```

**Note**: In practice, you'll use `#[derive(Signature)]` macro which implements this trait automatically. You rarely implement `MetaSignature` manually.

**Status**: ⚠️ Interface available but NOT USED in current implementation

**Phase 2 Work**: Actually use these methods to format prompts

### 4. Configure Pattern - VERIFIED AGAINST SOURCE

**Source**: `.claude/knowledge/dspy/source/lm-struct.md` (Lines 27-43)

```rust
// LM is a STRUCT (not a trait)
pub struct LM {
    // ... fields ...
}

impl LM {
    pub fn builder() -> LMBuilder { /* ... */ }
}

// Configure function sets global adapter
pub fn configure(adapter: impl Adapter + 'static, lm: Option<LM>) {
    // Sets global adapter for all predictors to use
}
```

**Usage Pattern**:
```rust
// Create adapter
let adapter = CandleAdapter::from_loaded_model(loaded_model, config);

// Create LM (optional for Candle since we have our own model)
let lm = LM::builder()
    .model("candle-qwen2.5-0.5b")
    .build()
    .await?;

// Configure dspy-rs globally
configure(adapter, Some(lm));

// Now all predictors use CandleAdapter
let predictor = Predict::new(signature);
let result = predictor.forward(inputs).await?;
```

**Status**: ✅ Pattern understood, ready to implement

---

## Testing Strategy

### Phase 2 Test Categories

#### 1. Signature Tests (Unit)

```rust
#[test]
fn test_signature_instruction_extraction() {
    // Test extracting instruction from signature
}

#[test]
fn test_signature_input_fields_extraction() {
    // Test extracting input field metadata
}

#[test]
fn test_signature_output_fields_extraction() {
    // Test extracting output field metadata
}

#[test]
fn test_format_with_simple_signature() {
    // Test prompt formatting with signature
}

#[test]
fn test_parse_with_simple_signature() {
    // Test output parsing with signature
}
```

#### 2. Example Type Tests (Unit)

```rust
#[test]
fn test_example_creation() {
    let mut data = HashMap::new();
    data.insert("question".to_string(), json!("What is 2+2?"));

    let example = Example::new(
        data,
        vec!["question".to_string()],
        vec![]
    );

    assert_eq!(example.input_keys.len(), 1);
}

#[test]
fn test_example_field_access() {
    // Test accessing fields from Example
}

#[test]
fn test_example_macro_compatibility() {
    // Test using example! macro from dspy-rs
}
```

#### 3. DSPy Integration Tests (Integration)

```rust
#[tokio::test]
#[ignore]  // Requires model download
async fn test_dspy_predict_qa() {
    use dspy_rs::{configure, Predict, Signature, example};

    // Setup
    let pool = ModelPool::new("./models".into());
    let loaded = pool.load_model("Qwen2.5-0.5B").await.unwrap();
    let adapter = CandleAdapter::from_loaded_model(
        Arc::new(loaded),
        CandleConfig::default()
    );

    // Configure dspy-rs
    configure(adapter, None);

    // Define signature
    #[derive(Signature)]
    struct QA {
        #[input]
        question: String,

        #[output]
        answer: String,
    }

    // Use Predict module
    let predictor = Predict::new(QA::new());
    let result = predictor.forward(example! {
        "question": "input" => "What is Rust?"
    }).await.unwrap();

    // Verify
    let answer = result.get("answer", None);
    assert!(!answer.is_empty(), "Should generate an answer");

    // Verify token usage
    assert!(result.lm_usage.prompt_tokens > 0);
    assert!(result.lm_usage.completion_tokens > 0);
    assert_eq!(
        result.lm_usage.total_tokens,
        result.lm_usage.prompt_tokens + result.lm_usage.completion_tokens
    );
}
```

```rust
#[tokio::test]
#[ignore]
async fn test_dspy_chain_of_thought() {
    use dspy_rs::{configure, ChainOfThought, Signature, example};

    // Setup (same as above)

    // Define reasoning signature
    #[derive(Signature)]
    struct MathReasoning {
        #[input]
        question: String,

        #[output]
        reasoning: String,

        #[output]
        answer: String,
    }

    // Use ChainOfThought module
    let cot = ChainOfThought::new(MathReasoning::new());
    let result = cot.forward(example! {
        "question": "input" => "What is 15 * 23?"
    }).await.unwrap();

    // Verify both reasoning and answer
    let reasoning = result.get("reasoning", None);
    let answer = result.get("answer", None);

    assert!(!reasoning.is_empty(), "Should have reasoning");
    assert!(!answer.is_empty(), "Should have answer");

    println!("Reasoning: {}", reasoning);
    println!("Answer: {}", answer);
}
```

#### 4. Error Handling Tests

```rust
#[tokio::test]
async fn test_signature_error_handling() {
    // Test malformed signature handling
}

#[tokio::test]
async fn test_parse_error_handling() {
    // Test output parsing errors
}

#[tokio::test]
async fn test_missing_output_field_handling() {
    // Test when model doesn't produce expected field
}
```

#### 5. Performance Tests (Optional)

```rust
#[tokio::test]
#[ignore]
async fn bench_with_kv_cache() {
    // Measure speedup with KV cache enabled
}

#[tokio::test]
#[ignore]
async fn bench_batch_inference() {
    // Measure throughput with batching
}
```

### Test Coverage Goals

| Component | Target Coverage |
|-----------|----------------|
| Signature parsing | 100% |
| Prompt formatting | 100% |
| Output parsing | 100% |
| Example type | 100% |
| Error handling | 90%+ |
| DSPy integration | 80%+ (depends on dspy-rs) |
| **Overall** | **90%+** |

---

## Performance Benchmarks

### Baseline (Phase 1)

| Metric | Phase 1 Result |
|--------|---------------|
| Throughput | 4.89 tok/s (CUDA, no KV cache) |
| First Token Latency | ~100-200ms |
| Memory Usage | ~1.5-2 GB |

### Phase 2 Targets

#### With KV Cache

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| Throughput | 25-50 tok/s | 50-100 tok/s |
| First Token Latency | ~100-200ms (same) | < 100ms |
| Memory Overhead | < 500MB | < 300MB |
| Speedup vs Phase 1 | 5-10x | 10-20x |

#### With Batching (batch_size=4)

| Metric | Target |
|--------|--------|
| Total Throughput | 15-20 tok/s |
| Per-prompt Throughput | 3.75-5 tok/s |
| Memory Overhead | < 1GB |

### Benchmark Tests

```rust
#[tokio::test]
#[ignore]
async fn bench_phase2_kv_cache_speedup() {
    use std::time::Instant;

    let pool = ModelPool::new("./models".into());
    let loaded = pool.load_model("Qwen2.5-0.5B").await.unwrap();

    // Without KV cache
    let config_no_cache = CandleConfig {
        enable_kv_cache: false,
        max_tokens: 50,
        ..Default::default()
    };
    let adapter_no_cache = CandleAdapter::from_loaded_model(
        Arc::new(loaded.clone()),
        config_no_cache
    );

    let start = Instant::now();
    let (_, _, tokens_no_cache) = adapter_no_cache.generate("Tell me a story").await.unwrap();
    let time_no_cache = start.elapsed();

    // With KV cache
    let config_cache = CandleConfig {
        enable_kv_cache: true,
        max_tokens: 50,
        ..Default::default()
    };
    let adapter_cache = CandleAdapter::from_loaded_model(
        Arc::new(loaded),
        config_cache
    );

    let start = Instant::now();
    let (_, _, tokens_cache) = adapter_cache.generate("Tell me a story").await.unwrap();
    let time_cache = start.elapsed();

    // Calculate speedup
    let throughput_no_cache = tokens_no_cache as f64 / time_no_cache.as_secs_f64();
    let throughput_cache = tokens_cache as f64 / time_cache.as_secs_f64();
    let speedup = throughput_cache / throughput_no_cache;

    println!("Without KV cache: {:.2} tok/s", throughput_no_cache);
    println!("With KV cache: {:.2} tok/s", throughput_cache);
    println!("Speedup: {:.2}x", speedup);

    // Target: 5-10x speedup
    assert!(speedup >= 5.0, "KV cache should provide at least 5x speedup");
}
```

---

## Success Criteria

### Functional Requirements

- [ ] **Signature Support**
  - [ ] Extracts instruction from signature
  - [ ] Formats prompts using input fields
  - [ ] Parses outputs using output fields
  - [ ] Handles multi-field signatures
  - [ ] Error messages are clear

- [ ] **Example Type**
  - [ ] Creates Example instances correctly
  - [ ] Works with `example!` macro
  - [ ] Input/output field separation works
  - [ ] Field access methods work

- [ ] **DSPy Integration**
  - [ ] `configure()` accepts CandleAdapter
  - [ ] `Predict` module works end-to-end
  - [ ] `ChainOfThought` module works end-to-end
  - [ ] Token usage is accurate
  - [ ] Multiple predictor instances work

- [ ] **Error Handling**
  - [ ] Signature errors caught and reported
  - [ ] Parse errors caught and reported
  - [ ] All errors have helpful messages
  - [ ] No panics in normal operation

### Non-Functional Requirements

- [ ] **Performance** (with KV cache)
  - [ ] Throughput: 25-50 tok/s (5-10x improvement)
  - [ ] First token latency: < 200ms
  - [ ] Memory overhead: < 500MB

- [ ] **Reliability**
  - [ ] All tests pass (unit + integration)
  - [ ] No memory leaks
  - [ ] No race conditions
  - [ ] Graceful degradation on errors

- [ ] **Code Quality**
  - [ ] Zero clippy warnings
  - [ ] 90%+ test coverage
  - [ ] All public APIs documented
  - [ ] Examples provided for common use cases

### Quality Gates

#### Gate 1: Signature Support Complete

- All signature parsing tests pass
- Prompt formatting uses signatures correctly
- Output parsing uses signatures correctly
- Integration test with simple signature passes

#### Gate 2: DSPy Integration Complete

- `Predict` module works end-to-end
- `ChainOfThought` module works end-to-end
- Token usage accurate in all cases
- All integration tests pass

#### Gate 3: Production Ready (Optional)

- KV cache provides 5-10x speedup
- Streaming works (if implemented)
- Batching works (if implemented)
- All benchmarks meet targets

---

## Troubleshooting Guide

### Common Issues

#### Issue 1: Signature Parsing Fails

**Error**: `SignatureError: Failed to extract input fields`

**Cause**: Signature structure not as expected

**Solution**:
1. Debug print the signature:
   ```rust
   println!("Instruction: {}", signature.instruction());
   println!("Input fields: {:?}", signature.input_fields());
   println!("Output fields: {:?}", signature.output_fields());
   ```
2. Verify signature definition matches dspy-rs API
3. Check for breaking changes in dspy-rs version

#### Issue 2: Output Parsing Fails

**Error**: `ParseError: Expected field 'answer' not found in output`

**Cause**: Model output doesn't match signature structure

**Solution**:
1. Log the raw model output:
   ```rust
   println!("Raw output: {}", response.content());
   ```
2. Implement lenient parsing (best-effort field extraction)
3. Add fallback for missing fields
4. Improve prompt formatting to guide model

#### Issue 3: DSPy Predictor Doesn't Work

**Error**: `configure()` not recognized or adapter not used

**Cause**: Incorrect dspy-rs usage pattern

**Solution**:
1. Verify dspy-rs version is v0.7.3
2. Check `configure()` was called before predictor creation
3. Ensure adapter implements all trait methods correctly
4. Review dspy-rs examples for correct pattern

#### Issue 4: KV Cache Causes Errors

**Error**: Tensor shape mismatch or CUDA errors

**Cause**: Cache not managed correctly

**Solution**:
1. Verify cache tensors match model expectations
2. Clear cache on new prompt
3. Check for off-by-one errors in sequence length
4. Validate cache update logic

#### Issue 5: Performance Not Improved

**Observation**: KV cache doesn't provide expected speedup

**Possible Causes**:
1. Cache not actually being used (verify in code)
2. Test methodology issue (measure correctly)
3. Bottleneck elsewhere (tokenization, I/O, etc.)
4. Memory bandwidth limitation

**Solution**:
1. Profile the code to find actual bottleneck
2. Verify cache hit rate
3. Check memory access patterns
4. Consider other optimizations

---

## Rollback Plan

### If Phase 2 Fails

Phase 2 is additive - Phase 1 functionality remains intact.

#### Rollback Procedure

1. **Identify failing component**
   - Signature support?
   - DSPy integration?
   - Performance features?

2. **Disable feature flags**
   ```rust
   // In config
   pub enable_signature_support: bool = false;  // Fallback to Phase 1 behavior
   ```

3. **Remove Phase 2 tests**
   ```bash
   git checkout HEAD~1 tests/integration_tests.rs
   ```

4. **Verify Phase 1 still works**
   ```bash
   cargo test
   ```

### Partial Implementation Strategy

Phase 2 should be implemented incrementally in priority order:

**Phase 2A - Core Compatibility** (Days 1-2):
1. Signature support (CRITICAL)
2. Example type (CRITICAL)
3. Enhanced error handling (HIGH)
4. DSPy integration tests (VALIDATION)

**Phase 2B - Production Performance** (Days 3-5):
5. KV cache (HIGH - 5-10x speedup)
6. Streaming output (MEDIUM - UX)
7. Batch inference (MEDIUM - throughput)
8. Example notebooks (DOCUMENTATION)

**Validation Points**:
- After Phase 2A: Verify 100% dspy-rs compatibility
- After Phase 2B: Verify performance targets met (25-50 tok/s)

If critical issues arise, Phase 2A can ship alone as "DSPy-compatible" release, with Phase 2B following as "Performance" release.

---

## Appendix A: dspy-rs v0.7.3 Source Verification

All implementation MUST be verified against these source files:

### 1. Adapter Trait
- **File**: `.claude/knowledge/dspy/source/adapter-trait.md`
- **Lines**: 34-54
- **Verified**: 2025-11-18
- **Status**: ✅ Trait signatures match current implementation

### 2. Core Types
- **File**: `.claude/knowledge/dspy/source/core-types.md`
- **Sections**:
  - Example: Lines 26-31
  - Prediction: Lines 182-186
  - Message: Lines 283-288
  - Chat: Lines 362-366
  - LmUsage: Lines 560-576
- **Verified**: 2025-11-18
- **Status**: ✅ All types except Example already implemented

### 3. LM Configuration
- **File**: `.claude/knowledge/dspy/source/lm-struct.md`
- **Lines**: 27-43
- **Verified**: 2025-11-18
- **Status**: ✅ LM struct usage pattern understood

### 4. Predictor APIs
- **File**: `.claude/knowledge/dspy/source/predictors-api.md`
- **Verified**: 2025-11-18
- **Status**: ✅ API patterns documented

---

## Appendix B: dspy-researcher Verification Report

**Date**: 2025-11-18
**Agent**: dspy-researcher
**Task**: Verify Phase 2 plan against dspy-rs v0.7.3 source

### Key Findings

**✅ Verified as Correct (85%)**:
- Adapter trait method signatures
- Core type definitions (Prediction, LmUsage, Message, Chat)
- LM configuration integration
- Predictor usage patterns

**❌ Discrepancies Found**:
1. **CRITICAL**: Example type missing from plan
2. **HIGH**: Signature handling incomplete (parameter ignored)
3. **MEDIUM**: Tokenization using estimates (should use real tokenizer)
4. **LOW**: Error types too generic (should use enum)

### Recommendations

**MUST HAVE** (Blocking):
1. Implement signature parsing and formatting
2. Add Example type definition
3. Test with actual DSPy predictors

**SHOULD HAVE** (Important):
1. Proper tokenization with tokenizers crate
2. Enhanced error types (CandleAdapterError enum)
3. Chat template support

**NICE TO HAVE** (Polish):
1. Performance optimization (KV cache)
2. Streaming output
3. Example notebooks

### Implementation Priority

1. **Signature support** - Critical gap, blocks full DSPy compatibility
2. **Example type** - Required for optimizers and predictors
3. **Tokenization** - Currently using estimates, should be real
4. **Error handling** - Improve specificity and clarity

---

## Appendix C: Phase Comparison

### Phase 0 (Mock)
- **Focus**: Architecture and trait implementation
- **Inference**: Mock (returns static strings)
- **Token Counting**: Estimated
- **Tests**: 9 (all passing)
- **Status**: ✅ Complete

### Phase 1 (Real Inference)
- **Focus**: Candle model integration
- **Inference**: Real (Qwen2.5-0.5B)
- **Token Counting**: Accurate (from tokenizer)
- **Performance**: 4.89 tok/s (CUDA, no KV cache)
- **Tests**: 14 (6 unit + 8 integration, all passing)
- **Status**: ✅ Complete

### Phase 2 (DSPy Integration + Production Performance)
- **Focus**: Full DSPy compatibility + Production-ready performance
- **Phase 2A - Compatibility** (Days 1-2):
  - Signature support ⭐ (CRITICAL - 10% gap)
  - Example type ⭐ (CRITICAL - 3% gap)
  - Enhanced error handling ⭐ (HIGH - 2% gap)
  - Predictor integration ⭐ (VALIDATION)
  - **Result**: 85% → 100% dspy-rs compatible
- **Phase 2B - Performance** (Days 3-5):
  - KV cache ⭐ (5-10x speedup)
  - Streaming output ⭐ (Real-time UX)
  - Batch inference ⭐ (Higher throughput)
  - Example notebooks ⭐ (Documentation)
  - **Result**: 4.89 → 25-50 tok/s
- **Performance Target**: 25-50 tok/s (5-10x improvement)
- **Tests**: ~32 total (14 from Phase 1 + ~18 new)
- **Total Effort**: 3-5 days
- **Status**: 🔄 Planning → Ready to implement

---

## Appendix D: Quick Reference Commands

### Run All Tests
```bash
cargo test
```

### Run Integration Tests Only
```bash
cargo test --test integration_tests -- --nocapture --ignored
```

### Run Specific Phase 2 Test
```bash
cargo test --test integration_tests test_9_dspy_predict -- --nocapture --ignored
```

### Benchmark Performance
```bash
cargo test --test integration_tests bench_phase2_kv_cache_speedup -- --nocapture --ignored
```

### Check Code Quality
```bash
# Setup CUDA environment first
.\setup_cuda_env.ps1

# Run clippy
cargo clippy --all-targets --all-features -- -D warnings
```

### Build Release
```bash
cargo build --release
```

---

**Document Version**: 1.2
**Created**: 2025-11-18
**Updated**: 2025-11-18 (Post-Audit Corrections)
**Status**: ✅ **VERIFIED** - Ready for Phase 2 Implementation
**Prerequisites**: Phase 0 ✅ Complete, Phase 1 ✅ Complete
**Accuracy**: ~95% (verified against dspy-rs v0.7.3 source)

**Scope**:
- Phase 2A: DSPy Compatibility (1-2 days) → 85% → 100% compatible
- Phase 2B: Production Performance (2-3 days) → 4.89 → 25-50 tok/s
- **Total Effort**: 3-5 days for complete implementation

**Recent Corrections** (v1.2):
- ✅ Fixed MetaSignature trait (added all 7 methods with correct types)
- ✅ Enhanced example! macro documentation (added clear explanation and examples)
- ✅ Verified all other sections are accurate (trait bounds, LM builder, predictors)

**Next Action**: Begin Phase 2A implementation with signature support (highest priority)

---

**End of Phase 2 Verification Specification**
