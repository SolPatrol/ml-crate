# dspy-rs LM Struct - Official Source Code

**Version**: 0.7.3
**Source**: https://github.com/krypticmouse/DSRs/blob/main/crates/dspy-rs/src/core/lm/mod.rs
**Date Verified**: 2025-11-16

---

## CRITICAL: LM is a Struct, Not a Trait

**There is NO `LanguageModel` trait in dspy-rs v0.7.3.**

Instead, dspy-rs provides an `LM` **struct** that wraps various provider clients (OpenAI, Anthropic, Gemini, etc.).

---

## LM Struct Definition

```rust
// From: crates/dspy-rs/src/core/lm/mod.rs

use bon::Builder;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::Mutex;

#[derive(Builder)]
pub struct LM {
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    #[builder(default = "openai:gpt-4o-mini".to_string())]
    pub model: String,
    #[builder(default = 0.7)]
    pub temperature: f32,
    #[builder(default = 512)]
    pub max_tokens: u32,
    #[builder(default = 10)]
    pub max_tool_iterations: u32,
    #[builder(default = false)]
    pub cache: bool,
    pub cache_handler: Option<Arc<Mutex<ResponseCache>>>,
    #[builder(skip)]
    client: Option<Arc<LMClient>>,
}
```

---

## Building an LM

```rust
// Option 1: OpenAI (default)
let lm = LM::builder()
    .model("gpt-4o-mini")
    .api_key(std::env::var("OPENAI_API_KEY")?)
    .temperature(0.7)
    .max_tokens(512)
    .build()
    .await?;

// Option 2: Anthropic
let lm = LM::builder()
    .model("anthropic:claude-3-5-sonnet-20241022")
    .api_key(std::env::var("ANTHROPIC_API_KEY")?)
    .build()
    .await?;

// Option 3: Local server (vLLM, text-generation-inference, etc.)
let lm = LM::builder()
    .base_url("http://localhost:8000/v1")
    .model("local-model")
    .build()
    .await?;

// Option 4: Custom OpenAI-compatible API
let lm = LM::builder()
    .base_url("https://api.custom.com/v1")
    .api_key("custom-key")
    .model("custom-model")
    .build()
    .await?;
```

---

## Main Method: `call()`

```rust
impl LM {
    pub async fn call(&self, messages: Chat, tools: Vec<Arc<dyn ToolDyn>>) -> Result<LMResponse> {
        // Uses rig crate's CompletionRequest under the hood
        // Handles tool calls, streaming, etc.
    }
}
```

**Parameters**:
- `messages: Chat` - The conversation history
- `tools: Vec<Arc<dyn ToolDyn>>` - Tool definitions for function calling

**Returns**: `LMResponse`

```rust
#[derive(Clone, Debug)]
pub struct LMResponse {
    /// Assistant message chosen by the provider
    pub output: Message,
    /// Token usage reported by the provider
    pub usage: LmUsage,
    /// Chat history including the freshly appended assistant response
    pub chat: Chat,
    /// Tool calls made by the provider
    pub tool_calls: Vec<ToolCall>,
    /// Tool executions made by the provider
    pub tool_executions: Vec<String>,
}
```

---

## How Adapters Use LM

The `Adapter` trait's `call()` method receives an `Arc<LM>`:

```rust
#[async_trait]
pub trait Adapter: Send + Sync + 'static {
    async fn call(
        &self,
        lm: Arc<LM>,  // ← LM is passed to the adapter
        signature: &dyn MetaSignature,
        inputs: Example,
        tools: Vec<Arc<dyn ToolDyn>>,
    ) -> Result<Prediction>;
}
```

**Adapters can:**
1. **Use the LM**: Call `lm.call(chat, tools)` to use a remote provider
2. **Ignore the LM**: Use their own model (e.g., Candle) and ignore the `lm` parameter

---

## Example: Using LM in an Adapter

```rust
// ChatAdapter uses the LM
impl Adapter for ChatAdapter {
    async fn call(
        &self,
        lm: Arc<LM>,  // Use this
        signature: &dyn MetaSignature,
        inputs: Example,
        tools: Vec<Arc<dyn ToolDyn>>,
    ) -> Result<Prediction> {
        let chat = self.format(signature, inputs);

        // Call the LM
        let response = lm.call(chat, tools).await?;

        let parsed = self.parse_response(signature, response.output);
        Ok(Prediction::new(parsed, response.usage))
    }
}
```

---

## Example: Ignoring LM in a Custom Adapter

```rust
// CandleAdapter ignores the LM
impl Adapter for CandleAdapter {
    async fn call(
        &self,
        _lm: Arc<LM>,  // Ignore this - we have our own model
        signature: &dyn MetaSignature,
        inputs: Example,
        _tools: Vec<Arc<dyn ToolDyn>>,
    ) -> Result<Prediction> {
        let chat = self.format(signature, inputs);

        // Use our own Candle model instead
        let response = self.run_candle_inference(&chat).await?;

        let parsed = self.parse_response(signature, response);
        Ok(Prediction::new(parsed, LmUsage::default()))
    }
}
```

---

## LMClient (Internal)

The `LM` struct wraps an internal `LMClient` enum that dispatches to provider-specific clients:

```rust
// From: crates/dspy-rs/src/core/lm/client_registry.rs

pub enum LMClient {
    OpenAI(openai::completion::CompletionModel),
    Anthropic(anthropic::completion::CompletionModel),
    Gemini(gemini::completion::CompletionModel),
    // ... other providers
}
```

This is implemented using the [`rig`](https://crates.io/crates/rig) crate for provider integrations.

---

## Configuration with Global Settings

dspy-rs uses global settings to configure the default adapter and LM:

```rust
use dspy_rs::{configure, ChatAdapter};

let adapter = ChatAdapter;
let lm = LM::builder()
    .model("gpt-4o-mini")
    .api_key(std::env::var("OPENAI_API_KEY")?)
    .build()
    .await?;

// Set global defaults
configure(adapter, Some(lm));
```

Now all predictors will use this adapter + LM by default:

```rust
let predictor = Predict::new(QuestionAnswer);
let result = predictor.forward(inputs).await?;  // Uses global adapter + LM
```

---

## DummyLM (For Testing)

dspy-rs also provides `DummyLM` for deterministic tests:

```rust
#[derive(Clone, Builder, Default)]
pub struct DummyLM {
    pub api_key: String,
    pub base_url: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub cache: bool,
    pub cache_handler: Option<Arc<Mutex<ResponseCache>>>,
}

impl DummyLM {
    pub async fn call(
        &self,
        example: Example,
        messages: Chat,
        prediction: String,  // ← You provide the response
    ) -> Result<LMResponse> {
        // Returns the prediction you provide without calling any API
    }
}
```

---

## Summary

**Key Facts about LM:**

1. ✅ `LM` is a **struct**, not a trait
2. ✅ It wraps provider clients (OpenAI, Anthropic, etc.) via the `rig` crate
3. ✅ Main method is `call(messages, tools) → LMResponse`
4. ✅ Adapters receive `Arc<LM>` but can ignore it
5. ✅ No `LanguageModel` trait exists in dspy-rs v0.7.3

**Do NOT:**
- ❌ Try to implement a `LanguageModel` trait (doesn't exist)
- ❌ Try to extend `LM` (it's a concrete struct)
- ❌ Confuse this with Python DSPy's LM class

**DO:**
- ✅ Use `LM::builder()` to create an LM instance
- ✅ Call `lm.call(chat, tools)` for inference
- ✅ Implement the `Adapter` trait to customize behavior
- ✅ Ignore the `lm` parameter if you have your own model

---

## References

- **Official Repo**: https://github.com/krypticmouse/DSRs
- **LM Struct**: `crates/dspy-rs/src/core/lm/mod.rs`
- **LMClient Enum**: `crates/dspy-rs/src/core/lm/client_registry.rs`
- **Rig Integration**: Uses https://crates.io/crates/rig for providers
- **Version**: 0.7.3 (verified 2025-11-16)
