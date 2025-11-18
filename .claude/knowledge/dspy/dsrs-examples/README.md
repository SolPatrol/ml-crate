# DSRs (dspy-rs) Rust Examples

This directory contains annotated examples from the DSRs repository demonstrating key patterns and capabilities.

## Available Examples

### 01-simple-qa-pipeline.rs
**Demonstrates**: Basic Signatures, Modules, Module composition, Chain-of-Thought
- Creates a QA system with answer rating
- Shows how to compose multiple predictors
- Uses `#[Signature(cot)]` for chain-of-thought reasoning

**Key Patterns**:
- Multiple signatures for different tasks
- Module composition (answerer + rater)
- Async forward() implementation
- Builder pattern for module construction

### 02-miprov2-optimization.rs
**Demonstrates**: MIPROv2 Optimizer, Evaluation, Data loading, Optimizable trait
- Loads data from HuggingFace
- Implements custom evaluation metrics
- Optimizes prompts using MIPROv2
- Compares baseline vs optimized performance

**Key Patterns**:
- `#[derive(Optimizable)]` enables optimization
- `#[parameter]` marks optimizable components
- Custom Evaluator trait for metrics
- DataLoader for datasets
- Baseline vs optimized comparison

### 03-module-iteration.rs
**Demonstrates**: Nested modules, Parameter iteration, Dynamic updates
- Shows how to iterate over module parameters
- Updates signature instructions dynamically
- Works with nested module structures

**Key Patterns**:
- Parameter iteration with `parameters()`
- Dynamic prompt modification
- Nested module handling
- Custom optimization strategies

## Full Example List from DSRs Repo

The complete DSRs repository contains 11 examples:

1. **01-simple.rs** - Basic introductory example
2. **02-module-iteration-and-updation.rs** - Working with module iteration and updates
3. **03-evaluate-hotpotqa.rs** - Evaluation using HotpotQA dataset
4. **04-optimize-hotpotqa.rs** - Optimization techniques for HotpotQA
5. **05-heterogenous-examples.rs** - Handling diverse example types
6. **06-other-providers-batch.rs** - Batch processing with alternative providers
7. **07-inspect-history.rs** - Inspection and history tracking
8. **08-optimize-mipro.rs** - MIPro optimization approach
9. **09-gepa-sentiment.rs** - GEPA implementation for sentiment analysis
10. **10-gepa-llm-judge.rs** - GEPA with LLM-based evaluation
11. **11-custom-client.rs** - Creating custom client implementations

**Repository**: https://github.com/krypticmouse/DSRs/tree/main/crates/dspy-rs/examples

## Running Examples

```bash
# Clone the DSRs repository
git clone https://github.com/krypticmouse/DSRs.git
cd DSRs

# Run any example
cargo run --example 01-simple
cargo run --example 08-optimize-mipro
```

## Common Patterns Across Examples

### Signature Definition
```rust
#[Signature]
struct TaskName {
    #[input]
    pub input_field: String,
    #[output]
    pub output_field: String,
}
```

### Module Implementation
```rust
#[derive(Builder, Optimizable)]
pub struct MyModule {
    #[parameter]
    pub predictor: Predict,
}

impl Module for MyModule {
    async fn forward(&self, input: Example) -> Result<Prediction> {
        self.predictor.forward(input).await
    }
}
```

### Configuration
```rust
configure(
    LM::builder()
        .model("openai:gpt-4o-mini".to_string())
        .build()
        .await?,
    ChatAdapter,
);
```

## Edge Deployment Considerations

For nano LLM integration:
- Use custom base URLs to point to local model servers
- Async design enables efficient concurrent operations
- Type-safe patterns reduce runtime errors
- Optimization happens at development time, not runtime
