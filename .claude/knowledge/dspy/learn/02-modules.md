# DSPy Modules Documentation

---

# ⚠️ WARNING: PYTHON REFERENCE ONLY - THIS PROJECT USES RUST ⚠️

**This file contains Python DSPy examples for conceptual learning.**

**For Rust implementation, see:**
- [../source/predictors-optimizers-evaluation.md](../source/predictors-optimizers-evaluation.md) - Rust Module trait and implementations
- [../source/adapter-trait.md](../source/adapter-trait.md) - How to create custom modules in Rust

**All code in this project MUST be Rust. Python examples below are for understanding concepts only.**

---

## Core Concept

A DSPy module is a fundamental building block for language model programs. According to the documentation: "A **DSPy module** is a building block for programs that use LMs." Each module abstracts a specific prompting technique while maintaining learnable parameters and supporting composition into larger programs.

## Key Module Types

**Built-in Modules:**
- **dspy.Predict**: Basic predictor handling instructions, demonstrations, and LM updates
- **dspy.ChainOfThought**: Enables step-by-step reasoning before generating responses
- **dspy.ProgramOfThought**: Generates executable code whose results inform outputs
- **dspy.ReAct**: Agent-based module enabling tool integration
- **dspy.MultiChainComparison**: Compares multiple outputs to produce final predictions

## Usage Pattern

Modules follow a three-step workflow:

1. **Declare** with a signature: `classify = dspy.Predict('sentence -> sentiment: bool')`
2. **Call** with inputs: `response = classify(sentence=sentence)`
3. **Access** outputs: `print(response.sentiment)`

Configuration options like `n=5` (multiple completions), `temperature`, and `max_len` can be passed during declaration.

## Composition

Modules compose naturally within Python classes inheriting from `dspy.Module`. The framework traces LM calls during compilation, enabling complex control flows. Custom modules implement a `forward()` method calling sub-modules freely without special syntax.

## Usage Tracking

Available in version 2.6.16+, enable tracking via: `dspy.configure(track_usage=True)`

Access statistics through: `usage = prediction_instance.get_lm_usage()`

Returns token counts (prompt, completion, total) and detailed breakdowns per language model.

---

## Rust DSRs Equivalent

In dspy-rs, modules implement the `Module` trait:

```rust
use dspy_rs::Module;
use async_trait::async_trait;

#[async_trait]
pub trait Module {
    async fn forward(&self, input: Example) -> Result<Prediction>;
}
```

**Built-in Predictors in DSRs:**
- `Predict` - Basic signature execution
- `ChainOfThought` - Multi-step reasoning
- `ReAct` - Reasoning + Action loops

**Custom Module Example:**
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

Key differences:
- Rust modules are async by default (using `tokio`)
- Type safety enforced at compile time
- Explicit error handling with `Result` types
