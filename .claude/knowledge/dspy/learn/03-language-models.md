# DSPy Language Models Documentation

---

# ⚠️ WARNING: PYTHON REFERENCE ONLY - THIS PROJECT USES RUST ⚠️

**This file contains Python DSPy examples for conceptual learning.**

**For Rust implementation, see:**
- [../source/lm-struct.md](../source/lm-struct.md) - Rust LM struct (NOT a trait!)
- [../source/adapter-trait.md](../source/adapter-trait.md) - How adapters use LM in Rust

**All code in this project MUST be Rust. Python examples below are for understanding concepts only.**

---

## Overview
DSPy provides a unified interface for configuring and using language models. The framework begins by setting up a default LM, such as OpenAI's GPT-4o-mini.

## Configuration
Basic setup involves creating an LM instance and configuring it globally:

```python
import dspy
lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_KEY')
dspy.configure(lm=lm)
```

## Supported Providers
DSPy supports numerous LM providers:

- **OpenAI**: GPT-4o-mini, GPT-3.5-turbo
- **Google**: Gemini models (via AI Studio)
- **Anthropic**: Claude Sonnet variants
- **Databricks**: Llama models
- **Local options**: SGLang, Ollama
- **Other providers**: Any service supported by LiteLLM (Anyscale, Together AI, Azure, AWS SageMaker, etc.)

Authentication typically uses environment variables (e.g., `OPENAI_API_KEY`) or direct API key parameters.

## Direct LM Calls
Language models can be invoked directly with unified syntax:

```python
lm("Say this is a test!", temperature=0.7)
lm(messages=[{"role": "user", "content": "Say this is a test!"}])
```

## Integration with DSPy Modules
The idiomatic approach uses DSPy modules with signatures:

```python
qa = dspy.ChainOfThought('question -> answer')
response = qa(question="Your question here?")
```

## Multi-LM Support
Switch between models using `dspy.configure()` globally or `dspy.context()` for scoped changes (thread-safe).

## Generation Parameters
Configurable options include temperature, max_tokens, and stop sequences, settable at initialization or per-call.

## Caching & Rollouts
Caching is enabled by default. Force new requests using unique `rollout_id` values with non-zero temperature.

## Metadata & History
Every LM maintains interaction history with metrics including prompts, outputs, token usage, costs, timestamps, and UUIDs accessible via `lm.history`.

## Advanced Features
- **Responses API**: Set `model_type="responses"` for models supporting enhanced reasoning endpoints
- **Custom LMs**: Inherit from `dspy.BaseLM` for custom implementations
- **Adapters**: Intermediate layer between signatures and LMs for advanced workflows

---

## Rust DSRs Equivalent

In dspy-rs, language models are configured similarly:

```rust
use dspy_rs::{configure, LM};
use dspy_rs::adapter::ChatAdapter;

configure(
    LM::builder()
        .model("gpt-4o-mini".to_string())
        .temperature(0.5)
        .build()
        .await?,
    ChatAdapter,
);
```

**Key Features in DSRs:**
- Async-first design using `tokio`
- Builder pattern for configuration
- Support for custom base URLs (enabling local models)
- Type-safe adapter system

**For Edge/Nano LLMs:**
```rust
// Configure with custom base URL for local model
LM::builder()
    .model("nano-llm".to_string())
    .base_url("http://localhost:8080".to_string())
    .build()
    .await?
```

This enables integration with locally-hosted nano LLMs for edge deployment scenarios.
