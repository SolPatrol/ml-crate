# ml-crate-dsrs

Embedded LLM inference library for Rust using DSPy-style modules and llama.cpp.

**License**: MIT

## Requirements

- **GPU Backend** (one of):
  - Vulkan SDK (default - works on AMD, NVIDIA, Intel)
  - CUDA 12.x toolkit (NVIDIA optimized)
  - Metal (macOS/Apple Silicon)
  - CPU fallback (no GPU required)
- Rust 1.70+
- GGUF model file: `qwen2.5-0.5b-instruct-q4_k_m.gguf`

## Quick Start

```rust
use ml_crate_dsrs::{ModelPool, LlamaCppAdapter, LlamaCppConfig, DSPyEngine, SignatureRegistry};
use dspy_rs::Signature;
use std::sync::Arc;
use std::path::PathBuf;

// 1. Define your signature
#[derive(Signature, Default)]
struct QA {
    #[input] question: String,
    #[output] answer: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 2. Load GGUF model
    let pool = ModelPool::new("./models".into());
    let loaded = pool.load_model("qwen2.5-0.5b-instruct-q4_k_m").await?;

    // 3. Create adapter
    let adapter = Arc::new(LlamaCppAdapter::from_loaded_model(
        loaded,
        LlamaCppConfig::default(),
    ));

    // 4. Register signatures
    let mut registry = SignatureRegistry::new();
    registry.register::<QA>("qa");

    // 5. Create engine
    let engine = DSPyEngine::new(
        PathBuf::from("./modules"),
        adapter,
        Arc::new(registry),
    ).await?;

    // 6. Invoke module
    let result = engine.invoke("qa.simple", serde_json::json!({
        "question": "What is Rust?"
    })).await?;

    println!("Answer: {}", result["answer"]);
    Ok(())
}
```

## SignatureRegistry

**You must register signatures before creating the engine.** The `signature_name` field in module JSON must match a registered signature.

```rust
use dspy_rs::Signature;
use ml_crate_dsrs::SignatureRegistry;

#[derive(Signature, Default)]
struct MySignature {
    #[input] query: String,
    #[output] response: String,
}

let mut registry = SignatureRegistry::new();
registry.register::<MySignature>("my.signature");
// Module JSON: "signature_name": "my.signature"
```

## Module Format

### manifest.json

```json
{
  "version": "1.0",
  "modules": {
    "qa.simple": {
      "path": "qa_simple.json",
      "hash": null,
      "tags": ["qa"]
    }
  }
}
```

### Module JSON (qa_simple.json)

```json
{
  "module_id": "qa.simple",
  "predictor_type": "predict",
  "signature_name": "qa",
  "instruction": "Answer the question concisely.",
  "demos": [
    {
      "inputs": { "question": "What is 2+2?" },
      "outputs": { "answer": "4" }
    }
  ],
  "tool_enabled": false,
  "metadata": {
    "optimizer": "manual",
    "version": "1.0.0"
  }
}
```

## Tool System

Register tools for ReAct-style modules:

```rust
use ml_crate_dsrs::inference::tools::Tool;
use async_trait::async_trait;

struct GetData;

#[async_trait]
impl Tool for GetData {
    fn name(&self) -> &str { "get_data" }
    fn description(&self) -> &str { "Retrieves data" }

    async fn execute(&self, args: serde_json::Value) -> Result<serde_json::Value, ml_crate_dsrs::inference::tools::ToolError> {
        Ok(serde_json::json!({ "result": "data" }))
    }
}

// Register and invoke
engine.register_tool(Arc::new(GetData)).await;
let result = engine.invoke_with_tools("tool.module", input).await?;
```

See [specs/02-dspy-engine.md](specs/02-dspy-engine.md) for tool-enabled module format.

## Rhai Integration

```rhai
// Invoke from Rhai script
let result = dspy_invoke("qa.simple", #{
    question: "What is Rust?"
});

print(result.answer);
```

See [specs/02-dspy-engine.md](specs/02-dspy-engine.md) for full Rhai API.

## Directory Structure

```
your-project/
├── models/
│   └── qwen2.5-0.5b-instruct-q4_k_m.gguf
└── modules/
    ├── manifest.json
    └── qa_simple.json
```

## Backend Selection

Build with different GPU backends:

```bash
# Vulkan (default) - AMD, NVIDIA, Intel GPUs
cargo build

# CUDA - NVIDIA optimized (+10-20% vs Vulkan)
cargo build --features cuda

# Metal - macOS/Apple Silicon
cargo build --features metal

# CPU only
cargo build --features cpu
```

## Documentation

- [Architecture Overview](specs/ARCH.md)
- [LlamaCpp Adapter Spec](specs/01-llamacpp-adapter.md)
- [DSPy Engine Spec](specs/02-dspy-engine.md)
