# dspy-rs Adapter Trait - Official Source Code

**Version**: 0.7.3
**Source**: https://github.com/krypticmouse/DSRs/blob/main/crates/dspy-rs/src/adapter/mod.rs
**Date Verified**: 2025-11-16

---

## CRITICAL: The Adapter Trait

This is the **ONLY** trait you need to implement for custom adapters in dspy-rs.

**There is NO `LanguageModel` trait in dspy-rs v0.7.3.**

---

## Trait Definition

```rust
// From: crates/dspy-rs/src/adapter/mod.rs

pub mod chat;

pub use chat::*;

use crate::{Chat, Example, LM, Message, MetaSignature, Prediction};
use anyhow::Result;
use async_trait::async_trait;
use rig::tool::ToolDyn;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

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

---

## Method Descriptions

### `format()`

**Purpose**: Convert a DSPy signature and input example into a Chat object.

**Parameters**:
- `signature: &dyn MetaSignature` - The DSPy signature defining input/output fields
- `inputs: Example` - The input values for this inference call

**Returns**: `Chat` - A sequence of Messages (System, User, Assistant)

**Example Implementation**:
```rust
fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat {
    let mut chat = Chat::new();

    // Add system message if signature has instruction
    if let Some(instruction) = signature.instruction() {
        chat.push_message(Message::system(instruction));
    }

    // Format input fields
    for (field_name, field_value) in inputs.iter() {
        chat.push_message(Message::user(&format!("{}: {}", field_name, field_value)));
    }

    chat
}
```

---

### `parse_response()`

**Purpose**: Extract output field values from the model's response.

**Parameters**:
- `signature: &dyn MetaSignature` - The signature defining expected outputs
- `response: Message` - The assistant's response message

**Returns**: `HashMap<String, Value>` - Map of output field names to values

**Example Implementation**:
```rust
fn parse_response(
    &self,
    signature: &dyn MetaSignature,
    response: Message,
) -> HashMap<String, Value> {
    let mut outputs = HashMap::new();

    let response_text = match response {
        Message::Assistant { content } => content,
        _ => String::new(),
    };

    // Simple: assign response to first output field
    if let Some(output_field) = signature.output_fields().first() {
        outputs.insert(output_field.clone(), Value::String(response_text));
    }

    outputs
}
```

---

### `call()`

**Purpose**: Main orchestration method - formats, runs inference, parses.

**Parameters**:
- `lm: Arc<LM>` - The language model (can be ignored if you have your own)
- `signature: &dyn MetaSignature` - The signature
- `inputs: Example` - Input values
- `tools: Vec<Arc<dyn ToolDyn>>` - Tool definitions (can be ignored for simple use)

**Returns**: `Result<Prediction>` - The prediction with outputs and usage stats

**Example Implementation**:
```rust
async fn call(
    &self,
    _lm: Arc<LM>,  // Ignore if using your own model
    signature: &dyn MetaSignature,
    inputs: Example,
    _tools: Vec<Arc<dyn ToolDyn>>,
) -> Result<Prediction> {
    // 1. Format
    let chat = self.format(signature, inputs);

    // 2. Run your model (e.g., Candle)
    let response_text = self.run_my_model(&chat).await?;

    // 3. Parse
    let response = Message::assistant(&response_text);
    let outputs = self.parse_response(signature, response);

    // 4. Return prediction
    Ok(Prediction::new(outputs, LmUsage::default()))
}
```

---

## Built-in Implementation: ChatAdapter

dspy-rs provides a reference implementation called `ChatAdapter`:

```rust
// From: crates/dspy-rs/src/adapter/chat.rs

pub struct ChatAdapter;

#[async_trait]
impl Adapter for ChatAdapter {
    fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat {
        // Implementation that formats signatures into chat format
        // ...
    }

    fn parse_response(
        &self,
        signature: &dyn MetaSignature,
        response: Message,
    ) -> HashMap<String, Value> {
        // Implementation that extracts fields from response
        // ...
    }

    async fn call(
        &self,
        lm: Arc<LM>,
        signature: &dyn MetaSignature,
        inputs: Example,
        tools: Vec<Arc<dyn ToolDyn>>,
    ) -> Result<Prediction> {
        let chat = self.format(signature, inputs);
        let response = lm.call(chat, tools).await?;
        let parsed = self.parse_response(signature, response.output);

        Ok(Prediction::new(parsed, response.usage))
    }
}
```

---

## How Predictors Use Adapters

```rust
// From: crates/dspy-rs/src/predictors/predict.rs

impl Predictor for Predict {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        let (adapter, lm) = {
            let settings = GLOBAL_SETTINGS.read().unwrap();
            (settings.adapter.clone(), settings.lm.clone())
        };

        // Adapter.call() is the main entry point
        adapter.call(lm, self.signature.as_ref(), inputs, self.tools.clone()).await
    }
}
```

**Flow**:
1. User calls `predictor.forward(inputs)`
2. Predictor gets adapter from global settings
3. Predictor calls `adapter.call(lm, signature, inputs, tools)`
4. Adapter formats → runs inference → parses
5. Returns `Prediction`

---

## Key Types

### Chat

```rust
// A sequence of messages
pub struct Chat {
    messages: Vec<Message>,
}

impl Chat {
    pub fn new() -> Self { /* ... */ }
    pub fn push_message(&mut self, msg: Message) { /* ... */ }
    pub fn messages(&self) -> &[Message] { /* ... */ }
}
```

### Message

```rust
pub enum Message {
    System { content: String },
    User { content: String },
    Assistant { content: String },
}

impl Message {
    pub fn system(content: &str) -> Self { /* ... */ }
    pub fn user(content: &str) -> Self { /* ... */ }
    pub fn assistant(content: &str) -> Self { /* ... */ }
}
```

### Example

```rust
// Input/output data for a signature
// Basically a HashMap<String, Value>
pub struct Example {
    data: HashMap<String, Value>,
}
```

### Prediction

```rust
pub struct Prediction {
    outputs: HashMap<String, Value>,
    usage: LmUsage,
}

impl Prediction {
    pub fn new(outputs: HashMap<String, Value>, usage: LmUsage) -> Self { /* ... */ }
}
```

### LmUsage

```rust
pub struct LmUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

impl Default for LmUsage {
    fn default() -> Self {
        Self { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    }
}
```

---

## Common Mistakes to Avoid

### ❌ WRONG: Looking for `LanguageModel` trait
```rust
// This trait DOES NOT EXIST in dspy-rs
impl LanguageModel for MyAdapter {
    async fn generate(&self, prompt: String, ...) -> Result<String> { /* ... */ }
}
```

### ✅ CORRECT: Implement `Adapter` trait
```rust
#[async_trait]
impl Adapter for MyAdapter {
    fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat { /* ... */ }
    fn parse_response(&self, signature: &dyn MetaSignature, response: Message) -> HashMap<String, Value> { /* ... */ }
    async fn call(&self, lm: Arc<LM>, signature: &dyn MetaSignature, inputs: Example, tools: Vec<Arc<dyn ToolDyn>>) -> Result<Prediction> { /* ... */ }
}
```

### ❌ WRONG: Template parameter
```rust
// There is NO template parameter
async fn call(&self, template: Option<String>, ...) { /* ... */ }
```

### ✅ CORRECT: No template parameter
```rust
// Templates are handled internally in format()
fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat { /* ... */ }
```

### ❌ WRONG: kwargs HashMap
```rust
// There is NO kwargs parameter
async fn call(&self, kwargs: &HashMap<String, Value>, ...) { /* ... */ }
```

### ✅ CORRECT: Parameters are in signature and inputs
```rust
// All parameters come through signature and inputs
async fn call(&self, lm: Arc<LM>, signature: &dyn MetaSignature, inputs: Example, tools: Vec<Arc<dyn ToolDyn>>) -> Result<Prediction>
```

---

## Configuration

To use a custom adapter:

```rust
use dspy_rs::configure;

let my_adapter = MyAdapter::new()?;
let lm = LM::builder().model("gpt-4").build().await?;

configure(my_adapter, Some(lm));
```

Or if your adapter has its own model (like Candle):

```rust
configure(my_adapter, None); // None = no LM needed
```

---

## Summary

**To implement a custom adapter for dspy-rs v0.7.3:**

1. ✅ Implement the `Adapter` trait
2. ✅ Implement 3 methods: `format()`, `parse_response()`, `call()`
3. ✅ Return `Chat` from `format()`
4. ✅ Return `HashMap<String, Value>` from `parse_response()`
5. ✅ Return `Result<Prediction>` from `call()`

**Do NOT:**
- ❌ Look for `LanguageModel` trait (doesn't exist)
- ❌ Add `template` parameter (doesn't exist)
- ❌ Add `kwargs` parameter (doesn't exist)
- ❌ Try to implement model weight optimization (dspy-rs optimizes prompts)

---

## References

- **Official Repo**: https://github.com/krypticmouse/DSRs
- **Adapter Trait**: `crates/dspy-rs/src/adapter/mod.rs`
- **ChatAdapter Example**: `crates/dspy-rs/src/adapter/chat.rs`
- **Predictor Usage**: `crates/dspy-rs/src/predictors/predict.rs`
- **Version**: 0.7.3 (verified 2025-11-16)
