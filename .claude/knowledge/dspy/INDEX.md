# dspy-rs v0.7.3 Knowledge Base - Complete Index

**Repository**: https://github.com/krypticmouse/DSRs
**Version**: 0.7.3
**Last Updated**: 2025-11-16
**Purpose**: Complete source code reference for dspy-rs Rust framework

---

# ⚠️ CRITICAL: THIS IS A RUST-ONLY CODEBASE ⚠️

**ALL IMPLEMENTATION MUST BE IN RUST**

This knowledge base contains:
- **`source/` directory**: ✅ **VERIFIED RUST CODE** from dspy-rs repository - **USE THIS FOR ALL IMPLEMENTATIONS**
- **`learn/` directory**: ⚠️ **PYTHON DSPy CONCEPTS FOR REFERENCE ONLY** - Explains framework philosophy, must be translated to Rust
- **`tutorials/` directory**: ⚠️ **PYTHON EXAMPLES FOR LEARNING** - Shows patterns, must be converted to Rust DSRs syntax

**NEVER use Python syntax in this project. All code must be Rust.**

---

## 🎯 Quick Start

If you're new to dspy-rs or auditing implementation plans, start here:

1. **[VERIFICATION.md](VERIFICATION.md)** - Why the previous audit was wrong and what was verified
2. **[source/adapter-trait.md](source/adapter-trait.md)** - The ONLY trait you need to implement for custom adapters
3. **[source/core-types.md](source/core-types.md)** - Core Rust types (Example, Prediction, Chat, Message)

---

## 📚 Knowledge Base Structure

```
.claude/knowledge/dspy/
├── INDEX.md                              (this file)
├── VERIFICATION.md                       Verification process and audit corrections
│
├── source/
│   ├── adapter-trait.md                  Adapter trait definition and examples
│   ├── lm-struct.md                      LM struct (NOT a trait!)
│   ├── core-types.md                     Example, Prediction, Chat, Message, etc.
│   └── predictors-optimizers-evaluation.md  Predictor, Optimizer, Evaluator traits
│
└── [future additions]
```

---

## 📖 Documentation Files

### 1. Verification & Source Code

#### [VERIFICATION.md](VERIFICATION.md)
**What it is**: Records the verification process that revealed the original audit was incorrect.

**Key Contents**:
- ❌ What was WRONG in the audit (LanguageModel trait, template parameter, kwargs parameter)
- ✅ What actually EXISTS (Adapter trait, LM struct)
- How to verify the source code yourself
- Update history

**When to use**:
- Before trusting ANY audit of dspy-rs
- When confused about what traits exist
- To understand why previous implementation plans were wrong

---

### 2. Core Traits & Structs

#### [source/adapter-trait.md](source/adapter-trait.md)
**What it is**: Complete documentation of the Adapter trait - the ONLY trait you implement for custom adapters.

**Key Contents**:
```rust
#[async_trait]
pub trait Adapter: Send + Sync + 'static {
    fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat;
    fn parse_response(&self, signature: &dyn MetaSignature, response: Message) -> HashMap<String, Value>;
    async fn call(&self, lm: Arc<LM>, signature: &dyn MetaSignature, inputs: Example, tools: Vec<Arc<dyn ToolDyn>>) -> Result<Prediction>;
}
```

**Includes**:
- Full trait definition from source
- Method descriptions with examples
- ChatAdapter reference implementation
- Common mistakes to avoid
- Configuration examples

**When to use**:
- Implementing a custom adapter (e.g., CandleAdapter)
- Understanding how adapters work
- Debugging adapter implementations

---

#### [source/lm-struct.md](source/lm-struct.md)
**What it is**: Documentation of the LM struct (clarifying it's NOT a trait).

**Key Contents**:
```rust
#[derive(Builder)]
pub struct LM {
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model: String,
    pub temperature: f32,
    pub max_tokens: u32,
    // ...
}
```

**Includes**:
- Full struct definition
- How to build LM instances
- Main method: `call(messages, tools) -> LMResponse`
- How adapters use (or ignore) LM
- DummyLM for testing

**When to use**:
- Understanding what LM is
- Building LM instances for different providers
- Understanding that adapters can ignore the LM parameter

---

#### [source/core-types.md](source/core-types.md)
**What it is**: Complete source code for all core data types.

**Key Contents**:
- **Example** - Input/output data container
- **Prediction** - Result container with usage stats
- **Message** - Chat message enum (System/User/Assistant)
- **Chat** - Sequence of messages
- **MetaSignature** - Trait for signatures (use `#[derive(Signature)]`)
- **LmUsage** - Token usage tracking
- **Macros** - `example!`, `prediction!`, `hashmap!`

**When to use**:
- Understanding the data flow in dspy-rs
- Working with Examples and Predictions
- Creating chat conversations
- Using the convenience macros

---

#### [source/predictors-optimizers-evaluation.md](source/predictors-optimizers-evaluation.md)
**What it is**: Documentation of the Predictor, Optimizer, and Evaluator traits.

**Key Contents**:
- **Predictor trait** - Forward inference interface
- **Predict struct** - Basic predictor implementation
- **Module trait** - Similar to Predictor with progress bars
- **Optimizable trait** - For modules that can be optimized
- **Optimizer trait** - MIPROv2, COPRO, GEPA
- **Evaluator trait** - For evaluation metrics
- **Global Settings** - configure(), get_lm(), GLOBAL_SETTINGS

**When to use**:
- Creating predictors
- Optimizing prompts
- Evaluating model performance
- Understanding the workflow from creation → optimization → evaluation

---

## 🔑 Key Concepts

### What IS in dspy-rs v0.7.3

✅ **Adapter trait** - For custom prompt formatting/parsing
✅ **LM struct** - Wraps provider clients (OpenAI, Anthropic, etc.)
✅ **Predictor trait** - For forward inference
✅ **Module trait** - Similar to Predictor with progress tracking
✅ **Optimizer trait** - For prompt optimization (MIPROv2, COPRO, GEPA)
✅ **Evaluator trait** - For metrics and evaluation
✅ **Optimizable trait** - For modules that can be optimized
✅ **MetaSignature trait** - For signature definitions (use derive macro)
✅ **Global settings** - configure(), get_lm()

### What is NOT in dspy-rs v0.7.3

❌ **LanguageModel trait** - DOES NOT EXIST
❌ **template parameter** - DOES NOT EXIST
❌ **kwargs parameter** - DOES NOT EXIST
❌ **generate() method** - Main method is `call()` on Adapter
❌ **Model weight optimization** - dspy-rs optimizes prompts, not weights

---

## 🛠️ Implementation Patterns

### Pattern 1: Custom Adapter (Recommended for Candle)

```rust
pub struct CandleAdapter {
    model: Arc<Mutex<Box<dyn candle_core::Module>>>,
    tokenizer: Arc<Mutex<tokenizers::Tokenizer>>,
    config: CandleConfig,
}

#[async_trait]
impl Adapter for CandleAdapter {
    fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat {
        // Convert signature + inputs to Chat
    }

    fn parse_response(&self, signature: &dyn MetaSignature, response: Message) -> HashMap<String, Value> {
        // Extract output fields from response
    }

    async fn call(&self, _lm: Arc<LM>, signature: &dyn MetaSignature, inputs: Example, _tools: Vec<Arc<dyn ToolDyn>>) -> Result<Prediction> {
        // 1. Format
        let chat = self.format(signature, inputs);

        // 2. Run Candle inference (ignore the lm parameter)
        let response_text = tokio::task::spawn_blocking(move || {
            // Candle inference here
        }).await??;

        // 3. Parse
        let response = Message::assistant(&response_text);
        let outputs = self.parse_response(signature, response);

        // 4. Return prediction
        Ok(Prediction::new(outputs, LmUsage::default()))
    }
}
```

**See**: [source/adapter-trait.md](source/adapter-trait.md) for full examples

---

### Pattern 2: Using dspy-rs (Rust)

```rust
use dspy_rs::{configure, ChatAdapter, LM, Predict, Signature, example};

// 1. Define signature
#[derive(Signature)]
struct QuestionAnswer {
    #[input]
    question: String,

    #[output]
    answer: String,
}

// 2. Configure global settings
let lm = LM::builder()
    .model("gpt-4o-mini")
    .api_key(std::env::var("OPENAI_API_KEY")?)
    .build()
    .await?;

configure(lm, ChatAdapter);

// 3. Create and use predictor
let predictor = Predict::new(QuestionAnswer::new());

let input = example! {
    "question": "input" => "What is 2+2?"
};

let result = predictor.forward(input).await?;
let answer = result.get("answer", None);
```

**See**: [source/predictors-optimizers-evaluation.md](source/predictors-optimizers-evaluation.md) for full workflow

---

## 🗂️ Source Code Reference

### Repository Structure

```
DSRs/
├── crates/
│   ├── dspy-rs/                    Main crate
│   │   ├── src/
│   │   │   ├── adapter/            ✅ Adapter trait & ChatAdapter
│   │   │   │   ├── mod.rs          → See: source/adapter-trait.md
│   │   │   │   └── chat.rs         → See: source/adapter-trait.md
│   │   │   ├── core/               ✅ Core types and traits
│   │   │   │   ├── lm/             → See: source/lm-struct.md
│   │   │   │   ├── module.rs       → See: source/predictors-optimizers-evaluation.md
│   │   │   │   ├── settings.rs     → See: source/predictors-optimizers-evaluation.md
│   │   │   │   └── signature.rs    → See: source/core-types.md
│   │   │   ├── data/               ✅ Data structures
│   │   │   │   ├── example.rs      → See: source/core-types.md
│   │   │   │   └── prediction.rs   → See: source/core-types.md
│   │   │   ├── evaluate/           ✅ Evaluation framework
│   │   │   │   └── evaluator.rs    → See: source/predictors-optimizers-evaluation.md
│   │   │   ├── optimizer/          ✅ Optimizers
│   │   │   │   ├── mod.rs          → See: source/predictors-optimizers-evaluation.md
│   │   │   │   ├── mipro.rs        MIPROv2
│   │   │   │   ├── copro.rs        COPRO
│   │   │   │   └── gepa.rs         GEPA
│   │   │   ├── predictors/         ✅ Predictors
│   │   │   │   ├── mod.rs          → See: source/predictors-optimizers-evaluation.md
│   │   │   │   └── predict.rs      → See: source/predictors-optimizers-evaluation.md
│   │   │   ├── utils/              Utilities
│   │   │   └── lib.rs              Public API & macros
│   │   └── Cargo.toml
│   └── dsrs-macros/                Derive macros (#[derive(Signature)])
└── Cargo.toml
```

---

## 🚨 Common Pitfalls

### Mistake 1: Looking for LanguageModel trait
**Wrong**: Trying to implement a `LanguageModel` trait
**Right**: Implement the `Adapter` trait

**See**: [VERIFICATION.md](VERIFICATION.md) for why this was wrong

---

### Mistake 2: Expecting template/kwargs parameters
**Wrong**: `async fn call(&self, template: Option<String>, kwargs: HashMap<String, Value>)`
**Right**: `async fn call(&self, lm: Arc<LM>, signature: &dyn MetaSignature, inputs: Example, tools: Vec<Arc<dyn ToolDyn>>)`

**See**: [source/adapter-trait.md](source/adapter-trait.md) for correct signature

---

### Mistake 3: Thinking you must use the LM parameter
**Wrong**: Assuming you must call `lm.call()`
**Right**: You can ignore `lm` and use your own model (e.g., Candle)

**See**: [source/adapter-trait.md](source/adapter-trait.md) and [source/lm-struct.md](source/lm-struct.md)

---

### Mistake 4: Confusing with Python DSPy
**Wrong**: Assuming Rust dspy-rs has the same API as Python DSPy
**Right**: They are different implementations with different architectures

**See**: [VERIFICATION.md](VERIFICATION.md) under "Common Confusion Sources"

---

### Mistake 5: Trying to optimize model weights
**Wrong**: Expecting dspy-rs to optimize model weights
**Right**: dspy-rs optimizes prompts and instructions, not model parameters

**See**: [source/predictors-optimizers-evaluation.md](source/predictors-optimizers-evaluation.md)

---

## 🔍 How to Find What You Need

### "I want to implement a Candle adapter"
→ Start with [source/adapter-trait.md](source/adapter-trait.md)
→ See example Rust implementation in CandleAdapter section
→ Reference [VERIFICATION.md](VERIFICATION.md) for verified trait signatures

### "I want to understand how LM works"
→ Read [source/lm-struct.md](source/lm-struct.md)
→ See how adapters use it in [source/adapter-trait.md](source/adapter-trait.md)

### "I want to create a predictor"
→ Read [source/predictors-optimizers-evaluation.md](source/predictors-optimizers-evaluation.md)
→ See Predict struct and Predictor trait

### "I want to understand Example/Prediction types"
→ Read [source/core-types.md](source/core-types.md)
→ See usage examples with macros

### "I want to optimize my prompts"
→ Read Optimizer section in [source/predictors-optimizers-evaluation.md](source/predictors-optimizers-evaluation.md)
→ See MIPROv2, COPRO, GEPA examples

### "I want to evaluate my model"
→ Read Evaluator section in [source/predictors-optimizers-evaluation.md](source/predictors-optimizers-evaluation.md)
→ Implement the `metric()` function

### "I don't trust the audit"
→ Read [VERIFICATION.md](VERIFICATION.md)
→ Follow the verification commands to check source yourself

---

## ✅ Verification Checklist

Before implementing based on this knowledge base, verify:

- [x] Version matches (currently 0.7.3)
- [x] Repository is https://github.com/krypticmouse/DSRs
- [x] Adapter trait exists in `crates/dspy-rs/src/adapter/mod.rs`
- [x] LM is a struct in `crates/dspy-rs/src/core/lm/mod.rs`
- [x] No LanguageModel trait exists
- [x] No template parameter exists
- [x] No kwargs parameter exists

**How to verify**: See [VERIFICATION.md](VERIFICATION.md) section "How to Verify"

---

## 📅 Version History

### 2025-11-16 (Initial Creation)
- Created comprehensive knowledge base
- Verified all source code against dspy-rs v0.7.3
- Documented Adapter, LM, Predictor, Optimizer, Evaluator
- Recorded verification process
- Corrected previous audit errors

---

## 🔗 External References

- **Official Repository**: https://github.com/krypticmouse/DSRs
- **Crate**: https://crates.io/crates/dspy-rs
- **Python DSPy** (different!): https://github.com/stanfordnlp/dspy
- **Rig Crate** (used internally): https://crates.io/crates/rig

---

## 📝 Notes for dspy-researcher Agent

**IMPORTANT**: When auditing implementation plans:

1. ✅ **Always reference this knowledge base first**
2. ✅ **Verify against actual source code when uncertain**
3. ✅ **Do NOT assume Python DSPy and Rust dspy-rs are the same**
4. ✅ **Check version compatibility (currently 0.7.3)**
5. ❌ **Do NOT invent traits or parameters that don't exist**
6. ❌ **Do NOT provide hypothetical implementations**

**If unsure**: Read [VERIFICATION.md](VERIFICATION.md) and follow the verification process.

---

## 🎓 Learning Path

**Beginner** (never used dspy-rs):
1. Read [ACTUAL_DSPY_RS_ARCHITECTURE.md](../../../specs/ACTUAL_DSPY_RS_ARCHITECTURE.md)
2. Skim [source/core-types.md](source/core-types.md)
3. Try the usage examples in [source/predictors-optimizers-evaluation.md](source/predictors-optimizers-evaluation.md)

**Intermediate** (implementing a custom adapter):
1. Read [source/adapter-trait.md](source/adapter-trait.md) thoroughly
2. Read [source/lm-struct.md](source/lm-struct.md)
3. Reference [ACTUAL_DSPY_RS_ARCHITECTURE.md](../../../specs/ACTUAL_DSPY_RS_ARCHITECTURE.md) for patterns

**Advanced** (optimizing and evaluating):
1. Read [source/predictors-optimizers-evaluation.md](source/predictors-optimizers-evaluation.md)
2. Study optimizer implementations in source repository
3. Implement custom Evaluator with metric functions

---

## 📧 Contributing

To add to this knowledge base:

1. Verify information against actual source code
2. Include source file locations
3. Provide working code examples
4. Update this INDEX.md with new files
5. Update version history

---

**Last Verified**: 2025-11-16
**dspy-rs Version**: 0.7.3
**Repository**: https://github.com/krypticmouse/DSRs
