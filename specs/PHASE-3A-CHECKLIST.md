# Phase 3A: DSPy Engine Core - Implementation Checklist

**Spec**: [08-dspy-engine.md](./08-dspy-engine.md)
**Status**: 📋 Not Started
**Target**: Core engine infrastructure for loading and executing pre-optimized DSPy modules

---

## Prerequisites

- [x] CandleAdapter Phase 2 complete (verified)
- [x] dspy-rs v0.7.3 API verified against knowledge base
- [x] `configure(lm, adapter)` pattern confirmed
- [ ] Model Pool available for CandleAdapter initialization

---

## Phase 3A Tasks

### 1. Project Structure Setup

- [ ] Create `src/inference/` directory
- [ ] Create `src/inference/mod.rs` with module exports
- [ ] Create `src/inference/error.rs` - DSPyEngineError enum
- [ ] Create `src/inference/module.rs` - OptimizedModule, Demo, SignatureDefinition
- [ ] Create `src/inference/manifest.rs` - ModuleManifest, ModuleEntry
- [ ] Create `src/inference/registry.rs` - SignatureRegistry
- [ ] Create `src/inference/engine.rs` - DSPyEngine struct
- [ ] Update `src/lib.rs` to export inference module

### 2. Error Types (`src/inference/error.rs`)

- [ ] Define `DSPyEngineError` enum with variants:
  - [ ] `ModuleNotFound(String)`
  - [ ] `SignatureNotFound(String)`
  - [ ] `ToolsNotEnabled(String)`
  - [ ] `MaxIterationsReached(usize)`
  - [ ] `InferenceError(String)`
  - [ ] `ParseError(String)`
  - [ ] `IoError(std::io::Error)`
  - [ ] `JsonError(serde_json::Error)`
  - [ ] `RuntimeError(String)`
- [ ] Implement `From` conversions for error types
- [ ] Add unit tests for error display

### 3. Module Types (`src/inference/module.rs`)

- [ ] Define `PredictorType` enum (Predict, ChainOfThought)
- [ ] Define `Demo` struct (inputs, outputs as HashMap<String, Value>)
- [ ] Define `FieldDefinition` struct (name, description, field_type)
- [ ] Define `SignatureDefinition` struct (inputs, outputs)
- [ ] Define `ModuleMetadata` struct (optimizer, optimized_at, metric_score, version)
- [ ] Define `OptimizedModule` struct with all fields from spec
- [ ] Add serde derives with `#[serde(rename_all = "snake_case")]` where needed
- [ ] Add `Default` impl for `ModuleMetadata`
- [ ] Write unit test: deserialize sample module JSON
- [ ] Write unit test: serialize/deserialize roundtrip

### 4. Manifest Types (`src/inference/manifest.rs`)

- [ ] Define `ModuleEntry` struct (path, hash, tags)
- [ ] Define `ModuleManifest` struct (version, modules HashMap)
- [ ] Add `Default` impl for `ModuleManifest`
- [ ] Implement `load_json<T>()` helper function
- [ ] Write unit test: deserialize sample manifest JSON
- [ ] Write unit test: empty manifest handling

### 5. Signature Registry (`src/inference/registry.rs`)

- [ ] Define `SignatureFactory` type alias
- [ ] Define `SignatureRegistry` struct with factories HashMap
- [ ] Implement `SignatureRegistry::new()`
- [ ] Implement `SignatureRegistry::register<S>(name)` generic method
- [ ] Implement `SignatureRegistry::create(name)` -> Option<Box<dyn MetaSignature>>
- [ ] Implement `SignatureRegistry::contains(name)` -> bool
- [ ] Implement `SignatureRegistry::names()` -> Vec<&str>
- [ ] Implement `Default` for `SignatureRegistry`
- [ ] Write unit test: register and create signature
- [ ] Write unit test: create non-existent signature returns None

### 6. Value ↔ Example Conversion Helpers

- [ ] Create `src/inference/conversion.rs`
- [ ] Implement `value_to_example(value: &Value, signature: &dyn MetaSignature)` -> Example
  - [ ] Use `signature.input_fields()` to get input key names
  - [ ] Use `signature.output_fields()` to get output key names (for demos)
  - [ ] Build `Example::new(data, input_keys, output_keys)`
- [ ] Implement `example_to_value(example: &Example)` -> Value
- [ ] Implement `prediction_to_value(prediction: &Prediction)` -> Value
- [ ] Write unit tests for each conversion function

### 7. DSPyEngine Core (`src/inference/engine.rs`)

- [ ] Define `DSPyEngine` struct with fields:
  - [ ] `modules: Arc<RwLock<HashMap<String, OptimizedModule>>>`
  - [ ] `manifest: Arc<RwLock<ModuleManifest>>`
  - [ ] `modules_dir: PathBuf`
  - [ ] `adapter: Arc<CandleAdapter>`
  - [ ] `signature_registry: Arc<SignatureRegistry>`
  - [ ] `runtime: Arc<tokio::runtime::Runtime>`
- [ ] Implement `DSPyEngine::new(modules_dir, adapter, signature_registry)`
  - [ ] Create LM with `LM::builder().model("local").build().await`
  - [ ] Call `dspy_rs::configure(lm, adapter.clone())`
  - [ ] Call `reload_all()` to load modules
- [ ] Implement `DSPyEngine::reload_all()` - load manifest and all modules
- [ ] Implement `DSPyEngine::reload_module(module_id)` - reload single module
- [ ] Implement `DSPyEngine::get_module(module_id)` -> Option<OptimizedModule>

### 8. Invoke Methods (`src/inference/engine.rs`)

- [ ] Implement `DSPyEngine::invoke(module_id, input)` -> Result<Value>
- [ ] Implement `DSPyEngine::invoke_sync(module_id, input)` - sync wrapper
- [ ] Implement `DSPyEngine::invoke_raw(module_id, input)` - internal dispatch
- [ ] Implement `DSPyEngine::execute_predict(module, input)`:
  - [ ] Get signature from registry: `let mut sig: Box<dyn MetaSignature> = registry.create(name)?`
  - [ ] **Mutate signature BEFORE creating predictor (use `sig.as_mut()`):**
    - [ ] Convert module.demos to Vec<Example> using conversion helper
    - [ ] Call `sig.as_mut().set_demos(demos)?`
    - [ ] Call `sig.as_mut().update_instruction(module.instruction.clone())?`
  - [ ] Build Example from input Value using conversion helper
  - [ ] **NOTE:** `Predict::new()` takes `impl MetaSignature`, but we have `Box<dyn MetaSignature>`
    - [ ] Option A: Modify SignatureRegistry to return concrete types (complex generics)
    - [ ] Option B: Create `Predict` directly with the boxed signature field
    - [ ] Option C: Use the internal `adapter.call()` directly, bypassing Predict struct
  - [ ] Call `predictor.forward(example).await`
  - [ ] Convert Prediction to Value and return
- [ ] Implement `DSPyEngine::execute_chain_of_thought(module, input)`:
  - [ ] Similar to execute_predict but with ChainOfThought predictor
  - [ ] ChainOfThought adds "reasoning" field automatically

### 9. Prompt Building Helpers

- [ ] Implement `build_prompt(module, input)` - format instruction + demos + input
- [ ] Implement `build_cot_prompt(module, input)` - include reasoning step
- [ ] Implement `parse_output(module, response)` - extract output fields
- [ ] Implement `parse_cot_output(module, response)` - extract reasoning + outputs

### 10. Integration Tests

- [ ] Create `tests/integration/dspy_engine_tests.rs`
- [ ] Test: Load engine with mock modules directory
- [ ] Test: invoke() with Predict module returns expected fields
- [ ] Test: invoke() with ChainOfThought module returns reasoning
- [ ] Test: Module not found error
- [ ] Test: Signature not found error
- [ ] Test: reload_module() updates module

### 11. Sample Module Files

- [ ] Create `tests/fixtures/modules/` directory
- [ ] Create `tests/fixtures/modules/manifest.json`
- [ ] Create `tests/fixtures/modules/test_predict.json` - simple Predict module
- [ ] Create `tests/fixtures/modules/test_cot.json` - ChainOfThought module

---

## Definition of Done

- [ ] All unit tests pass (`cargo test`)
- [ ] All integration tests pass
- [ ] No clippy warnings (`cargo clippy`)
- [ ] Code compiles with `--release`
- [ ] DSPyEngine can load modules from disk
- [ ] DSPyEngine.invoke() works with Predict modules
- [ ] DSPyEngine.invoke() works with ChainOfThought modules
- [ ] Value ↔ Example conversion works correctly

---

## Files to Create

```
src/
├── inference/
│   ├── mod.rs              # Module exports
│   ├── error.rs            # DSPyEngineError
│   ├── module.rs           # OptimizedModule, Demo, etc.
│   ├── manifest.rs         # ModuleManifest
│   ├── registry.rs         # SignatureRegistry
│   ├── conversion.rs       # Value ↔ Example helpers
│   └── engine.rs           # DSPyEngine
└── lib.rs                  # Update exports

tests/
├── integration/
│   └── dspy_engine_tests.rs
└── fixtures/
    └── modules/
        ├── manifest.json
        ├── test_predict.json
        └── test_cot.json
```

---

## Dependencies to Add

```toml
# Already in Cargo.toml from CandleAdapter
dspy-rs = "0.7.3"
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"
async-trait = "0.1"
tracing = "0.1"
```

---

## Notes

- Phase 3A focuses on core engine without tools or Rhai
- Tool system (ToolRegistry, ToolWrapper) is Phase 3B
- Rhai integration is Phase 3C
- Hot reload is Phase 3A-Optional (can defer)

---

## Progress Log

| Date | Task | Status | Notes |
|------|------|--------|-------|
| | | | |

