# dspy-rs Core Types - Official Source Code

**Version**: 0.7.3
**Source**: https://github.com/krypticmouse/DSRs/tree/main/crates/dspy-rs/src
**Date Verified**: 2025-11-16

---

## Overview

This document contains the actual source code for all core types in dspy-rs v0.7.3.

---

## Example

**Location**: `crates/dspy-rs/src/data/example.rs`

### Definition

```rust
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, ops::Index};

#[derive(Serialize, Deserialize, Default, Debug, Clone)]
pub struct Example {
    pub data: HashMap<String, Value>,
    pub input_keys: Vec<String>,
    pub output_keys: Vec<String>,
}
```

### Methods

```rust
impl Example {
    pub fn new(
        data: HashMap<String, Value>,
        input_keys: Vec<String>,
        output_keys: Vec<String>,
    ) -> Self {
        let output_keys = if !output_keys.is_empty() {
            output_keys
        } else if !input_keys.is_empty() {
            data.keys()
                .filter(|key| !input_keys.contains(key))
                .cloned()
                .collect()
        } else {
            vec![]
        };

        Self {
            data,
            input_keys,
            output_keys,
        }
    }

    pub fn get(&self, key: &str, default: Option<&str>) -> Value {
        self.data
            .get(key)
            .unwrap_or(&default.unwrap_or_default().to_string().into())
            .clone()
    }

    pub fn keys(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }

    pub fn values(&self) -> Vec<Value> {
        self.data.values().cloned().collect()
    }

    pub fn set_input_keys(&mut self, keys: Vec<String>) {
        self.input_keys = keys;

        self.output_keys = self
            .data
            .keys()
            .filter(|key| !self.input_keys.contains(key))
            .cloned()
            .collect();
    }

    pub fn with_input_keys(&self, keys: Vec<String>) -> Self {
        let output_keys = self
            .data
            .keys()
            .filter(|key| !keys.contains(key))
            .cloned()
            .collect();

        Self {
            data: self.data.clone(),
            input_keys: keys,
            output_keys,
        }
    }

    pub fn without(&self, keys: Vec<String>) -> Self {
        Self {
            data: self
                .data
                .iter()
                .filter(|(key, _)| !keys.contains(key))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            input_keys: self
                .input_keys
                .iter()
                .filter(|key| !keys.contains(key))
                .cloned()
                .collect(),
            output_keys: self
                .output_keys
                .iter()
                .filter(|key| !keys.contains(key))
                .cloned()
                .collect(),
        }
    }
}

impl IntoIterator for Example {
    type Item = (String, Value);
    type IntoIter = std::collections::hash_map::IntoIter<String, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl Index<String> for Example {
    type Output = Value;

    fn index(&self, index: String) -> &Self::Output {
        &self.data[&index]
    }
}
```

### Usage

```rust
use dspy_rs::{Example, hashmap};
use serde_json::json;

// Create an example with input and output fields
let example = Example::new(
    hashmap! {
        "question".to_string() => json!("What is 2+2?"),
        "answer".to_string() => json!("4")
    },
    vec!["question".to_string()],
    vec!["answer".to_string()],
);

// Or use the example! macro
let example = example! {
    "question": "input" => "What is 2+2?",
    "answer": "output" => "4"
};
```

---

## Prediction

**Location**: `crates/dspy-rs/src/data/prediction.rs`

### Definition

```rust
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, ops::Index};

use crate::LmUsage;

#[derive(Serialize, Deserialize, Default, Debug, Clone)]
pub struct Prediction {
    pub data: HashMap<String, serde_json::Value>,
    pub lm_usage: LmUsage,
}
```

### Methods

```rust
impl Prediction {
    pub fn new(data: HashMap<String, serde_json::Value>, lm_usage: LmUsage) -> Self {
        Self { data, lm_usage }
    }

    pub fn get(&self, key: &str, default: Option<&str>) -> serde_json::Value {
        self.data
            .get(key)
            .unwrap_or(&default.unwrap_or_default().to_string().into())
            .clone()
    }

    pub fn keys(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }

    pub fn values(&self) -> Vec<serde_json::Value> {
        self.data.values().cloned().collect()
    }

    pub fn set_lm_usage(&mut self, lm_usage: LmUsage) -> Self {
        self.lm_usage = lm_usage;
        self.clone()
    }
}

impl Index<String> for Prediction {
    type Output = serde_json::Value;

    fn index(&self, index: String) -> &Self::Output {
        &self.data[&index]
    }
}

impl IntoIterator for Prediction {
    type Item = (String, Value);
    type IntoIter = std::collections::hash_map::IntoIter<String, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl From<Vec<(String, Value)>> for Prediction {
    fn from(value: Vec<(String, Value)>) -> Self {
        Self {
            data: value.into_iter().collect(),
            lm_usage: LmUsage::default(),
        }
    }
}
```

### Usage

```rust
use dspy_rs::{Prediction, LmUsage, hashmap};
use serde_json::json;

// Create a prediction
let prediction = Prediction::new(
    hashmap! {
        "answer".to_string() => json!("4")
    },
    LmUsage {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
    },
);

// Or use the prediction! macro
let prediction = prediction! {
    "answer" => "4"
};

// Access fields
let answer = prediction.get("answer", None);
```

---

## Message

**Location**: `crates/dspy-rs/src/core/lm/chat.rs`

### Definition

```rust
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Message {
    System { content: String },
    User { content: String },
    Assistant { content: String },
}
```

### Methods

```rust
impl Message {
    pub fn new(role: &str, content: &str) -> Self {
        match role {
            "system" => Message::system(content),
            "user" => Message::user(content),
            "assistant" => Message::assistant(content),
            _ => panic!("Invalid role: {role}"),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Message::User {
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Message::Assistant {
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Message::System {
            content: content.into(),
        }
    }

    pub fn content(&self) -> String {
        match self {
            Message::System { content } => content.clone(),
            Message::User { content } => content.clone(),
            Message::Assistant { content } => content.clone(),
        }
    }

    pub fn to_json(&self) -> Value {
        match self {
            Message::System { content } => json!({ "role": "system", "content": content }),
            Message::User { content } => json!({ "role": "user", "content": content }),
            Message::Assistant { content } => json!({ "role": "assistant", "content": content }),
        }
    }
}
```

### Usage

```rust
use dspy_rs::Message;

let system_msg = Message::system("You are a helpful assistant");
let user_msg = Message::user("What is 2+2?");
let assistant_msg = Message::assistant("4");

// Get content
let text = user_msg.content(); // "What is 2+2?"
```

---

## Chat

**Location**: `crates/dspy-rs/src/core/lm/chat.rs`

### Definition

```rust
#[derive(Clone, Debug)]
pub struct Chat {
    pub messages: Vec<Message>,
}
```

### Methods

```rust
impl Chat {
    pub fn new(messages: Vec<Message>) -> Self {
        Self { messages }
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    pub fn push(&mut self, role: &str, content: &str) {
        self.messages.push(Message::new(role, content));
    }

    pub fn push_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn push_all(&mut self, chat: &Chat) {
        self.messages.extend(chat.messages.clone());
    }

    pub fn pop(&mut self) -> Option<Message> {
        self.messages.pop()
    }

    pub fn from_json(&self, json_dump: Value) -> Result<Self> {
        let messages = json_dump.as_array().unwrap();
        let messages = messages
            .iter()
            .map(|message| {
                Message::new(
                    message["role"].as_str().unwrap(),
                    message["content"].as_str().unwrap(),
                )
            })
            .collect();
        Ok(Self { messages })
    }

    pub fn to_json(&self) -> Value {
        let messages = self
            .messages
            .iter()
            .map(|message| message.to_json())
            .collect::<Vec<Value>>();
        json!(messages)
    }
}
```

### Usage

```rust
use dspy_rs::{Chat, Message};

// Create a new chat
let mut chat = Chat::new(vec![]);

// Add messages
chat.push_message(Message::system("You are a helpful assistant"));
chat.push_message(Message::user("What is 2+2?"));

// Or use push with role/content
chat.push("assistant", "4");

// Get all messages
for message in &chat.messages {
    println!("{}", message.content());
}
```

---

## MetaSignature

**Location**: `crates/dspy-rs/src/core/signature.rs`

### Definition

```rust
use crate::Example;
use anyhow::Result;
use serde_json::Value;

pub trait MetaSignature: Send + Sync {
    fn demos(&self) -> Vec<Example>;
    fn set_demos(&mut self, demos: Vec<Example>) -> Result<()>;
    fn instruction(&self) -> String;
    fn input_fields(&self) -> Value;
    fn output_fields(&self) -> Value;

    fn update_instruction(&mut self, instruction: String) -> Result<()>;
    fn append(&mut self, name: &str, value: Value) -> Result<()>;
}
```

### Usage

**You typically don't implement this manually.** Use the `#[derive(Signature)]` macro instead:

```rust
use dspy_rs::Signature;

#[derive(Signature)]
struct QuestionAnswer {
    #[input]
    question: String,

    #[output]
    answer: String,
}

// The macro generates the MetaSignature implementation
let signature = QuestionAnswer::new();
let instruction = signature.instruction();
let input_fields = signature.input_fields();
let output_fields = signature.output_fields();
```

---

## Predictor Trait

**Location**: `crates/dspy-rs/src/predictors/mod.rs`

### Definition

```rust
use crate::{Example, LM, Prediction};
use anyhow::Result;
use std::sync::Arc;

#[allow(async_fn_in_trait)]
pub trait Predictor: Send + Sync {
    async fn forward(&self, inputs: Example) -> anyhow::Result<Prediction>;

    async fn forward_with_config(
        &self,
        inputs: Example,
        lm: Arc<LM>
    ) -> anyhow::Result<Prediction>;

    async fn batch(&self, inputs: Vec<Example>) -> Result<Vec<Prediction>> {
        // Default implementation handles batching with concurrency
        // ...
    }

    async fn batch_with_config(
        &self,
        inputs: Vec<Example>,
        lm: Arc<LM>,
    ) -> Result<Vec<Prediction>> {
        // Default implementation handles batching with concurrency
        // ...
    }
}
```

### Usage

```rust
use dspy_rs::{Predictor, Predict, QuestionAnswer, example};

let predictor = Predict::new(QuestionAnswer::new());

let input = example! {
    "question": "input" => "What is 2+2?"
};

// Use the predictor
let prediction = predictor.forward(input).await?;
let answer = prediction.get("answer", None);
```

---

## LmUsage

**Location**: `crates/dspy-rs/src/core/lm/usage.rs`

### Definition

```rust
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Default, Debug, Clone)]
pub struct LmUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
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

impl Default for LmUsage {
    fn default() -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        }
    }
}
```

---

## Macros

**Location**: `crates/dspy-rs/src/lib.rs`

### example! macro

```rust
#[macro_export]
macro_rules! example {
    { $($key:literal : $field_type:literal => $value:expr),* $(,)? } => {{
        use std::collections::HashMap;
        use dspy_rs::data::example::Example;

        let mut input_keys = vec![];
        let mut output_keys = vec![];

        let mut fields = HashMap::new();
        $(
            if $field_type == "input" {
                input_keys.push($key.to_string());
            } else {
                output_keys.push($key.to_string());
            }

            fields.insert($key.to_string(), serde_json::to_value($value).unwrap());
        )*

        Example::new(
            fields,
            input_keys,
            output_keys,
        )
    }};
}
```

**Usage**:
```rust
let example = example! {
    "question": "input" => "What is 2+2?",
    "answer": "output" => "4"
};
```

### prediction! macro

```rust
#[macro_export]
macro_rules! prediction {
    { $($key:literal => $value:expr),* $(,)? } => {{
        use std::collections::HashMap;
        use dspy_rs::{Prediction, LmUsage};

        let mut fields = HashMap::new();
        $(
            fields.insert($key.to_string(), serde_json::to_value($value).unwrap());
        )*

        Prediction::new(fields, LmUsage::default())
    }};
}
```

**Usage**:
```rust
let prediction = prediction! {
    "answer" => "4",
    "confidence" => 0.95
};
```

### hashmap! macro

```rust
#[macro_export]
macro_rules! hashmap {
    () => {
        ::std::collections::HashMap::new()
    };

    ($($key:expr => $value:expr),+ $(,)?) => {
        ::std::collections::HashMap::from([ $(($key, $value)),* ])
    };
}
```

**Usage**:
```rust
let map = hashmap! {
    "key1" => "value1",
    "key2" => "value2"
};
```

---

## Summary

**Key Core Types**:

1. ✅ **Example** - Input/output data container with field tracking
2. ✅ **Prediction** - Result container with LM usage stats
3. ✅ **Message** - Chat message (System/User/Assistant)
4. ✅ **Chat** - Sequence of messages
5. ✅ **MetaSignature** - Trait for signature definitions (use `#[derive(Signature)]`)
6. ✅ **Predictor** - Trait for predictors (forward inference)
7. ✅ **LmUsage** - Token usage tracking

**Helper Macros**:
- `example!` - Create Example instances
- `prediction!` - Create Prediction instances
- `hashmap!` - Create HashMap instances

**Do NOT**:
- ❌ Manually implement MetaSignature (use `#[derive(Signature)]`)
- ❌ Create Message with invalid roles
- ❌ Confuse Example with Prediction

**DO**:
- ✅ Use the macros for cleaner code
- ✅ Track input_keys and output_keys properly in Example
- ✅ Include LmUsage in Prediction for token tracking
- ✅ Use Chat to build conversation histories

---

## References

- **Official Repo**: https://github.com/krypticmouse/DSRs
- **Example**: `crates/dspy-rs/src/data/example.rs`
- **Prediction**: `crates/dspy-rs/src/data/prediction.rs`
- **Chat/Message**: `crates/dspy-rs/src/core/lm/chat.rs`
- **Signature**: `crates/dspy-rs/src/core/signature.rs`
- **Predictor**: `crates/dspy-rs/src/predictors/mod.rs`
- **Macros**: `crates/dspy-rs/src/lib.rs`
- **Version**: 0.7.3 (verified 2025-11-16)
