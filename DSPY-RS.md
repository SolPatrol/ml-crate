# DSPy-RS (DSRs) Framework Documentation

## Overview

**DSRs** is a ground-up Rust rewrite of the DSPy framework, designed for building robust, high-performance applications powered by Language Models. Unlike a direct port, DSRs leverages Rust's type system, memory safety, and concurrency features.

**Current Version**: 0.7.3
**License**: Apache 2.0
**Repository**: https://github.com/krypticmouse/DSRs

## Philosophy

DSPy/DSRs is about **programming (not prompting)** language models. Instead of writing prompts, you define:
- **What** the task is (via Signatures)
- **How** to accomplish it (via Modules/Predictors)
- **How to optimize** it (via Optimizers)

## Core Architecture

### 1. Signatures

Type-safe contracts that define what the LLM should do.

```rust
use dspy_rs::Signature;

#[Signature]
struct SentimentAnalyzer {
    #[input]
    pub text: String,

    #[output]
    pub sentiment: String,
}
```

**Key Features**:
- Chain-of-thought support: `#[Signature(cot)]`
- Multiple inputs/outputs supported
- Strongly typed with compile-time checks

### 2. Predictors

Pre-built modules for LLM interaction:

- **Predict**: Basic signature execution
- **ChainOfThought**: Multi-step reasoning with intermediate steps
- **ReAct**: Reasoning + Action loops for agentic behavior

```rust
use dspy_rs::predictors::Predict;

let predictor = Predict::new(SentimentAnalyzer::new());
let result = predictor.forward(example).await?;
```

### 3. Modules

Composable pipeline components implementing the `Module` trait.

```rust
#[async_trait]
pub trait Module {
    async fn forward(&self, input: Example) -> Result<Prediction>;
}
```

Modules can be:
- Chained together for complex workflows
- Nested within each other
- Optimized independently

### 4. Language Models

Configurable LM backends with async operations.

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

**Supported Backends**:
- OpenAI API (via `OPENAI_API_KEY`)
- Custom backends (via base URL configuration)
- Local models (with appropriate adapters)

### 5. Data Structures

**Example**: Input data for predictions
```rust
let example = example! {
    "text": "input" => "This product is amazing!",
};
```

**Prediction**: Strongly-typed LLM outputs
```rust
let sentiment = result.get("sentiment", None);
```

### 6. Optimization

Two powerful optimizers for improving LLM performance:

**COPRO** (Collaborative Prompt Optimization):
- Iteratively refines prompts
- Uses LLM feedback for improvement
- Collaborative approach to optimization

**MIPROv2** (Multi-prompt Instruction Proposal Optimizer v2):
- Generates multiple prompt candidates
- Evaluates candidates on dev set
- Selects optimal prompts automatically

## Module Structure

The dspy-rs crate is organized into:

- **adapter**: Chat-based integrations and protocol conversions
- **core**: Foundational components and utilities
- **data**: Data handling and processing
- **evaluate**: Assessment and evaluation framework
- **optimizer**: COPRO, MIPROv2, GEPA optimization algorithms
- **predictors**: Predict, ChainOfThought, ReAct implementations
- **utils**: General utility functions

## Key Dependencies

- **Async Runtime**: `tokio`, `async-trait`
- **Serialization**: `serde`, `serde_json`
- **Data Processing**: `arrow`, `parquet`
- **ML Integration**: `rig-core`, `hf-hub`
- **Concurrency**: `rayon`, `futures`

## Three-Stage Development Process

### 1. Programming Stage

Define your task and design initial pipeline:
- Create Signatures for task specifications
- Choose appropriate Predictors
- Build Module compositions

### 2. Evaluation Stage

Measure system performance:
- Collect development dataset
- Define custom metrics via `Evaluator` trait
- Iterate based on evaluation results

### 3. Optimization Stage

Tune prompts and weights:
- Use COPRO or MIPROv2 optimizers
- Run optimization on dev set
- Validate improvements on test set

## Quick Start

**1. Add Dependency**

```toml
[dependencies]
dsrs = { package = "dspy-rs", version = "0.7.3" }
tokio = { version = "1", features = ["full"] }
```

**2. Set API Key**

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

**3. Basic Example**

```rust
use dspy_rs::prelude::*;
use dspy_rs::predictors::Predict;

#[Signature]
struct QA {
    #[input]
    pub question: String,

    #[output]
    pub answer: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Configure LM
    configure(
        LM::builder()
            .model("gpt-4o-mini".to_string())
            .build()
            .await?,
        ChatAdapter,
    );

    // Create predictor
    let predictor = Predict::new(QA::new());

    // Run prediction
    let example = example! {
        "question": "input" => "What is the capital of France?",
    };

    let result = predictor.forward(example).await?;
    println!("Answer: {}", result.get("answer", None));

    Ok(())
}
```

## Advanced Patterns

### Chain-of-Thought Reasoning

```rust
#[Signature(cot)]
struct ComplexReasoning {
    #[input]
    pub problem: String,

    #[output]
    pub solution: String,
}
```

### Custom Modules

```rust
struct CustomPipeline {
    step1: Predict<TaskA>,
    step2: Predict<TaskB>,
}

#[async_trait]
impl Module for CustomPipeline {
    async fn forward(&self, input: Example) -> Result<Prediction> {
        let intermediate = self.step1.forward(input).await?;
        self.step2.forward(intermediate.into()).await
    }
}
```

### Custom Evaluation

```rust
struct CustomMetric;

impl Evaluator for CustomMetric {
    fn evaluate(&self, prediction: &Prediction, gold: &Example) -> f64 {
        // Custom evaluation logic
        0.0
    }
}
```

## Edge Deployment Considerations

For nano LLM integration and edge deployment:

1. **Model Loading**: DSRs supports custom base URLs - can point to local model servers
2. **Async Design**: Efficient for concurrent requests without blocking
3. **Type Safety**: Compile-time guarantees reduce runtime errors
4. **Memory Efficiency**: Rust's ownership system enables fine-grained control
5. **Custom Adapters**: Can create adapters for non-standard LLM backends

## Resources

- **Documentation**: https://dsrs.herumbshandilya.com/
- **API Reference**: https://docs.rs/dspy-rs/latest/dspy_rs/
- **Examples**: https://github.com/krypticmouse/DSRs/tree/main/crates/dspy-rs/examples
- **Quickstart**: https://github.com/darinkishore/dsrs-quickstart
- **DSPy Learning**: https://dspy.ai/learn/

## Macros

Convenience macros for common operations:

- `hashmap!` - Create HashMaps from key-value pairs
- `field!` - Define signature fields
- `example!` - Create Example instances
- `prediction!` - Create Prediction instances
- `sign!` - Signature helpers
- `#[Signature]` - Attribute macro for signature structs
- `#[derive(Optimizable)]` - Enable optimization for custom types

## Community

- **GitHub**: https://github.com/krypticmouse/DSRs
- **Discord**: Check repository for invite
- **X (Twitter)**: Community updates and discussions

## Next Steps

1. Explore the [quickstart repository](https://github.com/darinkishore/dsrs-quickstart)
2. Read through [examples](https://github.com/krypticmouse/DSRs/tree/main/crates/dspy-rs/examples)
3. Review [API documentation](https://docs.rs/dspy-rs/latest/dspy_rs/)
4. Join community channels for support

---

**Note**: This documentation is current as of dspy-rs v0.7.3. Always check the official documentation for the latest updates.
