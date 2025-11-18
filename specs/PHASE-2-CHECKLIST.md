# Phase 2 Implementation Checklist

**Date Created**: 2025-11-18
**Last Updated**: 2025-11-18 (Enhanced with implementation examples)
**Status**: ✅ READY FOR IMPLEMENTATION
**Version**: dspy-rs v0.7.3
**Prerequisites**: Phase 0 ✅ Complete, Phase 1 ✅ Complete

---

## 📋 Verification Status

This checklist has been:
- ✅ **Cross-verified** against `PHASE-2-VERIFICATION.md`
- ✅ **Cross-verified** against `candle-adapter-implementation-plan-v7.md`
- ✅ **Cross-verified** against `01-candle-adapter.md`
- ✅ **Verified by dspy-researcher** against dspy-rs v0.7.3 source code
- ✅ **Enhanced** with implementation examples and detailed strategies

**Key Enhancements (v1.1)**:
- ✅ **Task 5**: Added demonstration formatting examples and strategy
- ✅ **Task 6**: Added parsing algorithm with 3-strategy approach (field markers, JSON, single-field)
- ✅ **Task 7**: Added 5 test categories with specific scenarios and edge cases
- ✅ **Architectural Clarity**: Documented that adapters format demos, not manage them

**Estimated Time Impact**: +4-7 hours for enhanced clarity (NOT +3-5 days as initially estimated)

---

## Overview

This checklist tracks the implementation of Phase 2, which adds full DSPy integration and production-ready performance features to the Candle Adapter.

### Phase 2 Goals

**Phase 2A: Core DSPy Compatibility** (85% → 100%)
- Close the 15% gap to achieve full dspy-rs v0.7.3 compatibility
- Estimated effort: 1-2 days
- Focus: Signature support, Example type, enhanced error handling

**Phase 2B: Production Performance** (100% → Production-Ready)
- Improve performance from 4.89 tok/s → 25-50 tok/s (5-10x speedup)
- Estimated effort: 2-3 days
- Focus: KV cache, streaming, batch inference

**Total Phase 2 Effort**: 3-5 days

---

## Phase 2A: Core DSPy Compatibility (Tasks 1-11)

### Task 1: Verify Current Implementation
**Priority**: CRITICAL
**Status**: ⬜ Not Started

**Description**: Review current adapter implementation against dspy-rs v0.7.3 source code

**Requirements**:
- [ ] Read current `src/adapters/candle/adapter.rs`
- [ ] Compare against `.claude/knowledge/dspy/source/adapter-trait.md`
- [ ] Verify method signatures match exactly
- [ ] Identify gaps in signature usage (currently parameter is ignored)
- [ ] Document what needs to be changed

**Success Criteria**:
- Clear understanding of what needs to be implemented
- List of specific changes required

**References**:
- `specs/PHASE-2-VERIFICATION.md` - Lines 134-161 (Adapter trait verification)
- `specs/candle-adapter-implementation-plan-v7.md` - Lines 99-122 (Adapter trait)
- `.claude/knowledge/dspy/source/adapter-trait.md` - Lines 34-54

---

### Task 2: Implement Example Type
**Priority**: CRITICAL (~3% of gap)
**Status**: ⬜ Not Started

**Description**: Define the `Example` struct that matches dspy-rs v0.7.3 exactly

**Requirements**:
- [ ] Create `Example` struct with 3 fields:
  ```rust
  pub struct Example {
      pub data: HashMap<String, Value>,      // All fields
      pub input_keys: Vec<String>,           // Which are inputs
      pub output_keys: Vec<String>,          // Which are outputs
  }
  ```
- [ ] Implement constructor: `Example::new(data, input_keys, output_keys)`
- [ ] Add helper methods: `inputs()`, `outputs()`, `get()`, `set()`
- [ ] Ensure it matches dspy-rs v0.7.3 exactly

**Success Criteria**:
- Example type compiles
- Matches source code structure exactly
- Helper methods work correctly

**References**:
- `specs/PHASE-2-VERIFICATION.md` - Lines 195-244 (Example type implementation)
- `specs/candle-adapter-implementation-plan-v7.md` - Lines 619-625
- `.claude/knowledge/dspy/source/core-types.md` - Lines 26-31

**Files to Create/Modify**:
- Possibly `src/adapters/candle/types.rs` or add to existing file

---

### Task 3: Create CandleAdapterError Enum
**Priority**: HIGH (~2% of gap)
**Status**: ⬜ Not Started

**Description**: Replace generic anyhow errors with specific error types

**Requirements**:
- [ ] Create `CandleAdapterError` enum with variants:
  - `InferenceFailed(String)`
  - `TokenizationFailed(String)`
  - `ContextTooLong { actual: usize, max: usize }`
  - `SignatureError(String)` - NEW for Phase 2
  - `ParseError(String)` - NEW for Phase 2
  - `ConfigError(String)`
  - `ModelLoadError(String)`
- [ ] Implement `Display` and `Error` traits
- [ ] Implement `From<CandleAdapterError>` for `anyhow::Error`
- [ ] Update all error handling to use specific types

**Success Criteria**:
- All error types compile
- Error messages are clear and actionable
- Error conversion works correctly
- All methods use specific error types (not generic anyhow)

**References**:
- `specs/PHASE-2-VERIFICATION.md` - Lines 246-268 (Error handling)
- `specs/candle-adapter-implementation-plan-v7.md` - Appendix A

**Files to Modify**:
- `src/adapters/candle/error.rs` (if exists, else create)
- `src/adapters/candle/adapter.rs` (update error handling)

---

### Task 4: Implement Signature Parsing
**Priority**: CRITICAL (~10% of gap)
**Status**: ⬜ Not Started

**Description**: Extract and use signature information (instruction, input_fields, output_fields)

**Requirements**:
- [ ] Create helper method `extract_signature_info(&self, signature: &dyn MetaSignature) -> SignatureInfo`
- [ ] Extract `instruction()` text from signature
- [ ] Extract `input_fields()` from signature (returns `Vec<Value>`)
- [ ] Extract `output_fields()` from signature (returns `Vec<Value>`)
- [ ] Parse field metadata (names, types, descriptions)
- [ ] Handle optional fields: `prefix()`, `desc()`, `signature_name()`
- [ ] Validate signature structure

**Success Criteria**:
- Can extract all signature information
- Handles all field types correctly
- Validates signature structure
- Works with MetaSignature trait

**References**:
- `specs/PHASE-2-VERIFICATION.md` - Lines 162-193 (Signature support)
- `specs/candle-adapter-implementation-plan-v7.md` - Lines 629-667
- `.claude/knowledge/dspy/source/core-types.md` - Lines 459-468 (MetaSignature trait)

**Files to Modify**:
- `src/adapters/candle/adapter.rs` (add helper methods)

---

### Task 5: Update format() Method
**Priority**: CRITICAL
**Status**: ⬜ Not Started

**Description**: Use signature instruction and input fields in prompt formatting

**Requirements**:
- [ ] Use `signature.instruction()` for system message
- [ ] Use `signature.input_fields()` to format user message
- [ ] Add output field hints to prompt (from `signature.output_fields()`)
- [ ] Format demonstrations if provided (from signature or inputs)
- [ ] Ensure Chat structure is correct

**Current Issue**:
The current implementation ignores most signature information. Need to actually use it.

**Implementation Strategy**:

The adapter receives demonstrations as part of the signature or as a parameter. The adapter's responsibility is to **format** demonstrations into the prompt, not to manage or store them.

**Example 1: Simple Question-Answer Signature**
```rust
// Signature definition (from dspy-rs)
#[derive(Signature)]
struct QuestionAnswer {
    #[input]
    question: String,

    #[output]
    answer: String,
}

// Expected format() output:
// System: Answer questions concisely and accurately.
//
// User:
// Question: What is the capital of France?
//
// Answer:
```

**Example 2: Signature with Demonstrations**
```rust
// When signature has demonstrations set via set_demos()
let signature = QuestionAnswer::new();
signature.set_demos(vec![
    example! {
        "question": "input" => "What is 2+2?",
        "answer": "output" => "4"
    },
    example! {
        "question": "input" => "What is the capital of Spain?",
        "answer": "output" => "Madrid"
    }
]);

// Expected format() output:
// System: Answer questions concisely and accurately.
//
// Demonstrations:
// Question: What is 2+2?
// Answer: 4
//
// Question: What is the capital of Spain?
// Answer: Madrid
//
// User:
// Question: What is the capital of France?
//
// Answer:
```

**Example 3: Multi-field Signature**
```rust
#[derive(Signature)]
struct MathReasoning {
    #[input]
    question: String,

    #[output]
    reasoning: String,

    #[output]
    answer: String,
}

// Expected format() output:
// System: Solve math problems with step-by-step reasoning.
//
// User:
// Question: What is 15 * 23?
//
// Reasoning:
// Answer:
```

**Implementation Notes**:
1. **Demonstrations are optional**: Check if signature has demos via `signature.demos()`
2. **Format demos using signature field definitions**: Use input/output field names from signature
3. **Demos come before the actual prompt**: They provide context/examples for the model
4. **Adapter doesn't manage demos**: Just formats what's passed in from signature/predictor

**Success Criteria**:
- System message includes instruction
- User message includes all input fields
- Output fields are hinted in prompt
- Demonstrations formatted correctly when present
- Works with 0, 1, or many demonstrations
- Field names from signature used in formatting

**References**:
- `specs/PHASE-2-VERIFICATION.md` - Lines 162-193 (Signature support)
- `specs/candle-adapter-implementation-plan-v7.md` - Lines 277-298
- `specs/01-candle-adapter.md` - Format method section
- `.claude/knowledge/dspy/source/adapter-trait.md` - Adapter trait definition

**Files to Modify**:
- `src/adapters/candle/adapter.rs` (update `format()` method)

---

### Task 6: Update parse_response() Method
**Priority**: CRITICAL
**Status**: ⬜ Not Started

**Description**: Parse response based on signature output fields

**Requirements**:
- [ ] Use `signature.output_fields()` to identify expected fields
- [ ] Extract each output field from response text
- [ ] Convert strings to appropriate Value types
- [ ] Handle missing fields gracefully (return ParseError)
- [ ] Support partial outputs (optional fields)
- [ ] Return `HashMap<String, Value>` with all extracted fields

**Current Issue**:
Current implementation just dumps full response into first field. Need structured parsing.

**Parsing Strategy**:

The model's completion text needs to be parsed to extract the structured output fields defined by the signature. The parsing strategy depends on how the model formats its output.

**Strategy 1: Field Marker Parsing (Recommended)**

Expect the model to output fields with markers like "FieldName: value".

**Example 1: Single Output Field**
```rust
// Signature
struct QA {
    #[input] question: String,
    #[output] answer: String,
}

// Model completion:
"Paris"
// OR
"Answer: Paris"

// Parsing logic:
// 1. Get output_fields from signature: ["answer"]
// 2. If text starts with "Answer:", extract after colon
// 3. Otherwise, treat entire text as the answer value
// 4. Return: { "answer": "Paris" }
```

**Example 2: Multi-field Output**
```rust
// Signature
struct MathReasoning {
    #[input] question: String,
    #[output] reasoning: String,
    #[output] answer: String,
}

// Model completion:
"Reasoning: To multiply 15 by 23, I can break it down: 15 * 20 = 300, 15 * 3 = 45, total = 345
Answer: 345"

// Parsing logic:
// 1. Get output_fields: ["reasoning", "answer"]
// 2. Use regex to find "Reasoning: (.*?)\nAnswer: (.*)"
// 3. Extract matches into HashMap
// 4. Return: {
//      "reasoning": "To multiply 15 by 23, I can break it down: 15 * 20 = 300, 15 * 3 = 45, total = 345",
//      "answer": "345"
//    }
```

**Strategy 2: JSON Parsing (Fallback)**

If the model outputs JSON, parse it directly.

```rust
// Model completion:
"{ \"reasoning\": \"...\", \"answer\": \"345\" }"

// Parsing logic:
// 1. Try to parse as JSON
// 2. If successful, extract output fields from JSON object
// 3. Return matched fields as HashMap
```

**Implementation Algorithm**:

```rust
fn parse_response(
    &self,
    signature: &dyn MetaSignature,
    response: Message,
) -> HashMap<String, Value> {
    let content = response.content();
    let output_fields = signature.output_fields();
    let mut results = HashMap::new();

    // Strategy 1: Try field marker parsing
    for field in output_fields.iter() {
        let field_name = field.as_str().unwrap(); // or field["name"]

        // Look for "FieldName: value" pattern
        let pattern = format!(r"{}:\s*(.+?)(?:\n|$)", field_name);
        if let Some(captures) = Regex::new(&pattern).captures(content) {
            results.insert(
                field_name.to_string(),
                Value::String(captures[1].trim().to_string())
            );
        }
    }

    // Strategy 2: If no fields found, try JSON parsing
    if results.is_empty() {
        if let Ok(json) = serde_json::from_str::<Value>(content) {
            for field in output_fields.iter() {
                let field_name = field.as_str().unwrap();
                if let Some(value) = json.get(field_name) {
                    results.insert(field_name.to_string(), value.clone());
                }
            }
        }
    }

    // Strategy 3: Single field - use entire response
    if results.is_empty() && output_fields.len() == 1 {
        let field_name = output_fields[0].as_str().unwrap();
        results.insert(field_name.to_string(), Value::String(content.to_string()));
    }

    results
}
```

**Error Handling**:

```rust
// If required fields are missing after parsing:
let missing: Vec<_> = output_fields
    .iter()
    .filter(|f| !results.contains_key(f.as_str().unwrap()))
    .collect();

if !missing.is_empty() {
    return Err(CandleAdapterError::ParseError(
        format!("Missing output fields: {:?}", missing)
    ));
}
```

**Success Criteria**:
- Extracts all output fields correctly using field markers
- Falls back to JSON parsing if markers not found
- Handles single-field case (entire response is the value)
- Handles missing fields with clear errors
- Returns proper HashMap structure
- Works with multi-field signatures
- Regex patterns are efficient and correct

**References**:
- `specs/PHASE-2-VERIFICATION.md` - Lines 162-193 (Signature support)
- `specs/candle-adapter-implementation-plan-v7.md` - Lines 300-322
- `specs/01-candle-adapter.md` - Parse response section
- `.claude/knowledge/dspy/source/adapter-trait.md` - parse_response method

**Files to Modify**:
- `src/adapters/candle/adapter.rs` (update `parse_response()` method)
- Add `regex` crate dependency to `Cargo.toml` if not already present

---

### Task 7: Write Unit Tests for Signature Parsing
**Priority**: HIGH
**Status**: ⬜ Not Started

**Description**: Test signature extraction and usage

**Requirements**:
- [ ] Test: Extract instruction from signature
- [ ] Test: Extract input fields from signature
- [ ] Test: Extract output fields from signature
- [ ] Test: Format prompt with signature
- [ ] Test: Parse response with signature
- [ ] Test: Handle invalid signature structure

**Specific Test Scenarios**:

**Test Category 1: Signature Field Extraction**
```rust
#[test]
fn test_extract_signature_fields() {
    // Test extracting input_fields() from signature
    // Test extracting output_fields() from signature
    // Test extracting instruction() from signature
    // Verify field names, types, descriptions are correct
}
```

**Test Category 2: Prompt Formatting**
```rust
#[test]
fn test_format_simple_qa_signature() {
    // QA signature: question -> answer
    // Verify system message contains instruction
    // Verify user message contains "Question: {value}"
    // Verify prompt ends with "Answer:" hint
}

#[test]
fn test_format_multi_field_signature() {
    // Math signature: question -> reasoning, answer
    // Verify both output fields hinted in prompt
    // Verify correct field order
}

#[test]
fn test_format_with_demonstrations() {
    // Signature with 2 demonstrations set
    // Verify demos formatted before main prompt
    // Verify demo format: "Question: X\nAnswer: Y"
    // Test with 0, 1, and 5 demonstrations
}

#[test]
fn test_format_without_instruction() {
    // Signature with no instruction() set
    // Verify no system message in output
    // Verify formatting still works
}
```

**Test Category 3: Response Parsing**
```rust
#[test]
fn test_parse_single_field_response() {
    // Response: "Paris"
    // Signature has 1 output field: "answer"
    // Verify entire response extracted as answer value
}

#[test]
fn test_parse_field_marker_response() {
    // Response: "Answer: Paris"
    // Verify "Paris" extracted (not "Answer: Paris")
}

#[test]
fn test_parse_multi_field_response() {
    // Response: "Reasoning: ...\nAnswer: 345"
    // Verify both fields extracted correctly
    // Verify multi-line reasoning field handled
}

#[test]
fn test_parse_json_response() {
    // Response: "{ \"answer\": \"Paris\" }"
    // Verify JSON parsing fallback works
}

#[test]
fn test_parse_missing_field_error() {
    // Response: "Answer: Paris" (missing "reasoning" field)
    // Signature expects: reasoning, answer
    // Verify ParseError with clear message
}
```

**Test Category 4: Edge Cases**
```rust
#[test]
fn test_empty_demonstration_list() {
    // Signature with empty demos vec
    // Verify no demonstration section in prompt
}

#[test]
fn test_demonstration_formatting_edge_cases() {
    // Demo with empty field values
    // Demo with special characters
    // Demo with very long text
}

#[test]
fn test_response_parsing_edge_cases() {
    // Empty response
    // Response with only whitespace
    // Response with field marker but no value
    // Response with malformed JSON
}

#[test]
fn test_multi_turn_conversation() {
    // Multiple format() calls with same signature
    // Verify state doesn't leak between calls
    // Verify demonstrations consistent across calls
}
```

**Test Category 5: Integration with dspy-rs Types**
```rust
#[tokio::test]
async fn test_with_predict_module() {
    // Create QA signature
    // Setup CandleAdapter with mock model
    // Use dspy-rs Predict module
    // Verify full workflow: format -> generate -> parse
    // Verify output HashMap has expected fields
}

#[tokio::test]
async fn test_with_chain_of_thought() {
    // Multi-field signature (reasoning + answer)
    // Use ChainOfThought module
    // Verify both fields extracted
}

#[tokio::test]
async fn test_demonstration_injection() {
    // Create signature with demonstrations
    // Verify demos appear in formatted prompt
    // Verify model sees examples before query
}
```

**Success Criteria**:
- All signature parsing tests pass
- Edge cases covered (empty, malformed, missing)
- Clear test names and assertions
- Tests use `#[tokio::test]` for async operations
- Integration tests verify full workflow
- Demonstration formatting tested with 0, 1, 5+ demos
- Response parsing tested with multiple strategies
- Error cases return clear, actionable messages

**References**:
- `specs/PHASE-2-VERIFICATION.md` - Lines 719-746 (Signature tests)
- `.claude/knowledge/dspy/source/adapter-trait.md` - Adapter trait
- `.claude/knowledge/dspy/source/predictors-api.md` - Predict/ChainOfThought usage

**Files to Create/Modify**:
- `src/adapters/candle/adapter.rs` (add unit tests in tests module)
- `tests/integration_tests.rs` (add integration tests with dspy-rs modules)

---

### Task 8: Write Unit Tests for Example Type
**Priority**: HIGH
**Status**: ⬜ Not Started

**Description**: Test Example type creation and usage

**Requirements**:
- [ ] Test: Example creation (manual)
- [ ] Test: Example creation with `example!` macro (if available)
- [ ] Test: Field access methods (`inputs()`, `outputs()`, `get()`, `set()`)
- [ ] Test: Input/output key separation
- [ ] Test: Integration with format() method

**Success Criteria**:
- All Example tests pass
- Verifies correct structure
- Tests helper methods

**References**:
- `specs/PHASE-2-VERIFICATION.md` - Lines 748-775 (Example tests)
- `specs/PHASE-2-VERIFICATION.md` - Lines 209-239 (example! macro)

**Files to Create/Modify**:
- `tests/unit_tests.rs` or add to adapter tests

---

### Task 9: Write Integration Test with Predict Module
**Priority**: CRITICAL (Validation)
**Status**: ⬜ Not Started

**Description**: Test end-to-end with dspy-rs Predict module

**Requirements**:
- [ ] Create test with Predict module
- [ ] Define test signature (e.g., QuestionAnswer)
- [ ] Setup adapter with Model Pool
- [ ] Call `configure(adapter, None)`
- [ ] Use `Predict::new(signature)`
- [ ] Call `predictor.forward(inputs)`
- [ ] Verify output fields populated correctly
- [ ] Verify token usage tracked accurately

**Success Criteria**:
- Predict module works end-to-end
- Output fields are correct
- Token usage is accurate
- Test passes with real model

**References**:
- `specs/PHASE-2-VERIFICATION.md` - Lines 777-823 (DSPy Predict test)
- `specs/candle-adapter-implementation-plan-v7.md` - Lines 631-668 (Example 1)

**Files to Create/Modify**:
- `tests/integration_tests.rs` (add `test_9_dspy_predict_integration`)

---

### Task 10: Write Integration Test with ChainOfThought Module
**Priority**: HIGH (Validation)
**Status**: ⬜ Not Started

**Description**: Test reasoning with ChainOfThought module

**Requirements**:
- [ ] Create test with ChainOfThought module
- [ ] Define reasoning signature (e.g., MathReasoning with reasoning + answer fields)
- [ ] Setup adapter with Model Pool
- [ ] Use `ChainOfThought::new(signature)`
- [ ] Call `cot.forward(inputs)`
- [ ] Verify reasoning field extracted
- [ ] Verify answer field extracted
- [ ] Test multi-step reasoning

**Success Criteria**:
- ChainOfThought module works end-to-end
- Both reasoning and answer fields populated
- Multi-step reasoning works
- Test passes with real model

**References**:
- `specs/PHASE-2-VERIFICATION.md` - Lines 825-862 (ChainOfThought test)
- `specs/candle-adapter-implementation-plan-v7.md` - Lines 670-705 (Example 2)

**Files to Create/Modify**:
- `tests/integration_tests.rs` (add `test_11_chain_of_thought`)

---

### Task 11: Verify Phase 2A Completion
**Priority**: CRITICAL
**Status**: ⬜ Not Started

**Description**: Ensure 100% dspy-rs compatibility achieved

**Requirements**:
- [ ] Run all unit tests: `cargo test --lib`
- [ ] Run all integration tests: `cargo test --test integration_tests -- --ignored`
- [ ] Verify zero clippy warnings: `cargo clippy -- -D warnings`
- [ ] Check test coverage (target: 90%+)
- [ ] Verify against dspy-rs v0.7.3 source code
- [ ] Document any remaining gaps

**Success Criteria**:
- All tests pass (unit + integration)
- Zero clippy warnings
- 100% dspy-rs compatibility confirmed
- Ready to move to Phase 2B

**References**:
- `specs/PHASE-2-VERIFICATION.md` - Lines 999-1028 (Success criteria)
- `specs/PHASE-2-VERIFICATION.md` - Lines 1050-1056 (Quality Gate 1)

**Phase 2A Result**: ✅ 85% → 100% dspy-rs compatible

---

## Phase 2B: Production Performance (Tasks 12-18)

### Task 12: Implement KV Cache Structure
**Priority**: HIGH (5-10x speedup)
**Status**: ⬜ Not Started

**Description**: Add KV cache for faster token generation

**Requirements**:
- [ ] Create KV cache structure:
  - `key_cache: Vec<Tensor>` (one per layer)
  - `value_cache: Vec<Tensor>` (one per layer)
- [ ] Implement cache lifecycle management
- [ ] Add memory estimation and limits
- [ ] Pass cache to `model.forward()`
- [ ] Update cache after each token
- [ ] Clear cache on new prompt
- [ ] Reuse cache for continuation

**Success Criteria**:
- Cache structure compiles
- Cache updates correctly
- Memory overhead acceptable (< 500MB)
- No correctness regression

**References**:
- `specs/PHASE-2-VERIFICATION.md` - Lines 299-325 (KV cache implementation)

**Files to Modify**:
- `src/adapters/candle/adapter.rs` (add cache management)

---

### Task 13: Add KV Cache Configuration
**Priority**: HIGH
**Status**: ⬜ Not Started

**Description**: Add configuration options for KV cache

**Requirements**:
- [ ] Add to `CandleConfig`:
  - `enable_kv_cache: bool` (default: `true`)
  - `max_cache_size: Option<usize>` for memory limit
- [ ] Add cache statistics tracking
- [ ] Validate configuration options
- [ ] Update Default impl

**Success Criteria**:
- Configuration options work
- Defaults are sensible
- Cache can be disabled/enabled

**References**:
- `specs/PHASE-2-VERIFICATION.md` - Lines 366-379 (Configuration enhancements)
- `specs/candle-adapter-implementation-plan-v7.md` - Lines 443-475

**Files to Modify**:
- `src/adapters/candle/config.rs`

---

### Task 14: Benchmark KV Cache Performance
**Priority**: HIGH (Validation)
**Status**: ⬜ Not Started

**Description**: Measure speedup with KV cache enabled

**Requirements**:
- [ ] Create benchmark test
- [ ] Measure throughput WITHOUT KV cache (baseline)
- [ ] Measure throughput WITH KV cache
- [ ] Calculate speedup ratio
- [ ] Verify target: 5-10x speedup
- [ ] Measure memory overhead
- [ ] Document results

**Success Criteria**:
- Speedup is 5-10x (target: 4.89 → 25-50 tok/s)
- Memory overhead < 500MB
- No correctness regression
- Benchmark test passes

**References**:
- `specs/PHASE-2-VERIFICATION.md` - Lines 912-995 (Performance benchmarks)
- `specs/PHASE-2-VERIFICATION.md` - Lines 943-994 (Benchmark test code)

**Files to Create/Modify**:
- `tests/benchmarks.rs` or add to integration tests

---

### Task 15: Implement Streaming Output
**Priority**: MEDIUM (Better UX)
**Status**: ⬜ Not Started

**Description**: Add token-by-token streaming for real-time applications

**Requirements**:
- [ ] Create stream interface: `impl Stream<Item = Result<String>>`
- [ ] Yield tokens as generated
- [ ] Handle errors in stream
- [ ] Graceful stream termination
- [ ] Add configuration: `enable_streaming: bool`
- [ ] Test with async streams
- [ ] Ensure compatibility with DSPy predictors

**Success Criteria**:
- Streaming works correctly
- Tokens emitted in real-time
- Error handling is graceful
- Optional (doesn't break existing API)

**References**:
- `specs/PHASE-2-VERIFICATION.md` - Lines 327-345 (Streaming output)

**Files to Modify**:
- `src/adapters/candle/adapter.rs` (add streaming method)
- `src/adapters/candle/config.rs` (add streaming config)

---

### Task 16: Implement Batch Inference
**Priority**: MEDIUM (Higher throughput)
**Status**: ⬜ Not Started

**Description**: Support batching multiple requests for higher throughput

**Requirements**:
- [ ] Implement batching logic:
  - Pad sequences to same length
  - Single forward pass for batch
  - Unbatch outputs correctly
  - Handle variable-length sequences
- [ ] Add configuration: `batch_size: Option<usize>`
- [ ] Support dynamic vs. static batching
- [ ] Add timeout settings for batch collection
- [ ] Benchmark performance improvement

**Success Criteria**:
- Batching works correctly
- Higher GPU utilization
- Throughput improvement measured
- Latency impact acceptable

**References**:
- `specs/PHASE-2-VERIFICATION.md` - Lines 347-365 (Batch inference)

**Files to Modify**:
- `src/adapters/candle/adapter.rs` (add batching logic)
- `src/adapters/candle/config.rs` (add batch config)

---

### Task 17: Create Example Notebooks
**Priority**: MEDIUM (Documentation)
**Status**: ⬜ Not Started

**Description**: Create example notebooks demonstrating usage

**Requirements**:
- [ ] Example 1: Simple Q&A with Predict
- [ ] Example 2: Chain of Thought reasoning
- [ ] Example 3: Multi-turn conversation
- [ ] Example 4: Using KV cache for performance
- [ ] Example 5: Streaming output demo
- [ ] Example 6: Batch inference demo
- [ ] Each example includes:
  - Clear explanation
  - Complete working code
  - Expected output
  - Performance notes

**Success Criteria**:
- All examples run successfully
- Clear and well-documented
- Cover major use cases

**References**:
- `specs/candle-adapter-implementation-plan-v7.md` - Lines 630-707 (Complete examples)

**Files to Create**:
- `examples/01_simple_qa.rs`
- `examples/02_chain_of_thought.rs`
- `examples/03_multi_turn.rs`
- `examples/04_kv_cache.rs`
- `examples/05_streaming.rs`
- `examples/06_batching.rs`

---

### Task 18: Run Final Performance Benchmarks
**Priority**: HIGH (Validation)
**Status**: ⬜ Not Started

**Description**: Verify all performance targets are met

**Requirements**:
- [ ] Benchmark throughput with KV cache (target: 25-50 tok/s)
- [ ] Benchmark first token latency (target: < 200ms)
- [ ] Measure memory usage (target: < 2.5GB total)
- [ ] Benchmark batch inference throughput
- [ ] Compare against Phase 1 baseline (4.89 tok/s)
- [ ] Document all results
- [ ] Create performance report

**Success Criteria**:
- Throughput: 25-50 tok/s (5-10x improvement) ✅
- First token latency: < 200ms ✅
- Memory overhead: < 500MB ✅
- All targets met or exceeded

**References**:
- `specs/PHASE-2-VERIFICATION.md` - Lines 912-995 (Performance benchmarks)
- `specs/PHASE-2-VERIFICATION.md` - Lines 1030-1043 (Non-functional requirements)

**Phase 2B Result**: ✅ 4.89 tok/s → 25-50 tok/s (5-10x speedup)

---

## Phase 2 Completion (Tasks 19-20)

### Task 19: Run All Tests and Quality Checks
**Priority**: CRITICAL
**Status**: ⬜ Not Started

**Description**: Final validation before declaring Phase 2 complete

**Requirements**:
- [ ] Run all unit tests: `cargo test --lib`
- [ ] Run all integration tests: `cargo test --test integration_tests -- --ignored`
- [ ] Run all benchmarks: `cargo test --test benchmarks -- --ignored`
- [ ] Run clippy: `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] Check code coverage (target: 90%+)
- [ ] Run release build: `cargo build --release`
- [ ] Verify no memory leaks
- [ ] Verify no race conditions

**Success Criteria**:
- All tests pass ✅
- Zero clippy warnings ✅
- 90%+ test coverage ✅
- Clean release build ✅

**References**:
- `specs/PHASE-2-VERIFICATION.md` - Lines 1042-1070 (Success criteria & quality gates)
- `specs/PHASE-2-VERIFICATION.md` - Lines 1322-1356 (Quick reference commands)

---

### Task 20: Update Documentation and Create Completion Report
**Priority**: HIGH
**Status**: ⬜ Not Started

**Description**: Document Phase 2 completion and results

**Requirements**:
- [ ] Update README.md with Phase 2 features
- [ ] Update CHANGELOG.md
- [ ] Create Phase 2 completion report:
  - Summary of changes
  - Performance improvements achieved
  - Test results
  - Known limitations
  - Next steps (if any)
- [ ] Update all API documentation
- [ ] Create migration guide (if breaking changes)
- [ ] Update Phase 2 verification spec status

**Success Criteria**:
- All documentation updated
- Completion report created
- Changes well-documented
- Users can understand new features

**Files to Create/Modify**:
- `README.md`
- `CHANGELOG.md`
- `specs/PHASE-2-COMPLETION-REPORT.md` (new)
- `specs/PHASE-2-VERIFICATION.md` (update status)

---

## Progress Tracking

### Phase 2A: Core DSPy Compatibility
- **Tasks**: 1-11
- **Status**: ⬜ Not Started (0/11 complete)
- **Target**: 100% dspy-rs compatibility
- **Estimated Effort**: 1-2 days

### Phase 2B: Production Performance
- **Tasks**: 12-18
- **Status**: ⬜ Not Started (0/7 complete)
- **Target**: 5-10x performance improvement
- **Estimated Effort**: 2-3 days

### Phase 2 Completion
- **Tasks**: 19-20
- **Status**: ⬜ Not Started (0/2 complete)
- **Estimated Effort**: 0.5 days

### Overall Phase 2 Progress
- **Total Tasks**: 20
- **Completed**: 0
- **In Progress**: 0
- **Not Started**: 20
- **Overall Progress**: 0%

---

## Key Metrics

### Code Changes Expected
- **Files to Modify**: ~5-7 files
- **Files to Create**: ~3-5 files
- **New Functions**: ~15-20 functions
- **New Tests**: ~18-20 tests
- **Code Growth**: ~600-800 lines (from ~1,120 → ~1,720-1,920 lines)

### Performance Targets
- **Baseline (Phase 1)**: 4.89 tok/s
- **Target (Phase 2B)**: 25-50 tok/s
- **Speedup**: 5-10x
- **Memory Overhead**: < 500MB

### Quality Targets
- **Test Coverage**: 90%+
- **Clippy Warnings**: 0
- **Integration Tests**: All passing
- **dspy-rs Compatibility**: 100%

---

## References

### Specification Documents
- `specs/PHASE-2-VERIFICATION.md` - Complete Phase 2 requirements and verification
- `specs/candle-adapter-implementation-plan-v7.md` - Implementation plan v7.0
- `specs/01-candle-adapter.md` - Original adapter specification v0.5.0

### dspy-rs v0.7.3 Source Code
- `.claude/knowledge/dspy/source/adapter-trait.md` - Adapter trait definition
- `.claude/knowledge/dspy/source/core-types.md` - Core types (Example, Prediction, etc.)
- `.claude/knowledge/dspy/source/lm-struct.md` - LM struct and configuration
- `.claude/knowledge/dspy/source/predictors-api.md` - Predictor APIs

### Previous Phases
- Phase 0: ✅ Complete (Mock implementation, trait structure)
- Phase 1: ✅ Complete (Real Candle inference, token counting)

---

## Notes

### Critical Implementation Guidelines
1. **Always verify against dspy-rs v0.7.3 source** - Don't assume, check the actual code
2. **Prediction has ONLY 2 fields** - `data` and `lm_usage` (no other fields!)
3. **LmUsage uses u64** - Not usize, not u32
4. **Example type is required** - For full DSPy compatibility
5. **Signature must be used** - Currently it's ignored, Phase 2A fixes this

### Common Pitfalls to Avoid
- ❌ Don't invent fields in Prediction (e.g., `output`, `reasoning`, `raw`)
- ❌ Don't ignore signature parameter
- ❌ Don't use generic errors (use CandleAdapterError)
- ❌ Don't estimate token counts (use real tokenizer)

### Phase 2 Philosophy
- **Phase 2A**: Make it correct (100% dspy-rs compatible)
- **Phase 2B**: Make it fast (production-ready performance)
- **Quality**: Maintain high standards (90%+ coverage, zero warnings)

---

**Document Version**: 1.0
**Created**: 2025-11-18
**Last Updated**: 2025-11-18
**Status**: 📋 Ready for Review and Verification
**Next Action**: Have checklist reviewed by dspy-researcher and cross-referenced with spec docs
