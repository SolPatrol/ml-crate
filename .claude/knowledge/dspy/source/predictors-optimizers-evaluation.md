# dspy-rs Predictors, Optimizers, and Evaluation - Official Source Code

**Version**: 0.7.3
**Source**: https://github.com/krypticmouse/DSRs/tree/main/crates/dspy-rs/src
**Date Verified**: 2025-11-16

---

## Overview

This document contains the actual source code for the Predictor trait, Module trait, Optimizer trait, and Evaluator trait in dspy-rs v0.7.3.

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
    /// Forward pass - uses global settings for adapter and LM
    async fn forward(&self, inputs: Example) -> anyhow::Result<Prediction>;

    /// Forward pass with custom LM configuration
    async fn forward_with_config(
        &self,
        inputs: Example,
        lm: Arc<LM>
    ) -> anyhow::Result<Prediction>;

    /// Batch forward pass (default implementation with concurrency)
    async fn batch(&self, inputs: Vec<Example>) -> Result<Vec<Prediction>> {
        let indexed_results: Vec<(usize, Result<Prediction>)> =
            stream::iter(inputs.into_iter().enumerate())
                .map(|(idx, input)| async move {
                    let result = self.forward(input).await;
                    (idx, result)
                })
                .buffer_unordered(32) // MAX_CONCURRENCY
                .collect()
                .await;

        // Sort results back to original order
        let mut indexed_results = indexed_results;
        indexed_results.sort_by_key(|(idx, _)| *idx);

        // Collect predictions and handle errors
        let mut predictions = Vec::with_capacity(indexed_results.len());
        for (_, result) in indexed_results {
            predictions.push(result?);
        }
        Ok(predictions)
    }

    /// Batch forward pass with custom LM configuration
    async fn batch_with_config(
        &self,
        inputs: Vec<Example>,
        lm: Arc<LM>,
    ) -> Result<Vec<Prediction>> {
        // Similar implementation to batch() but uses forward_with_config()
        // ...
    }
}
```

---

## Predict Struct

**Location**: `crates/dspy-rs/src/predictors/predict.rs`

### Definition

```rust
use indexmap::IndexMap;
use rig::tool::ToolDyn;
use std::sync::Arc;

use crate::core::{MetaSignature, Optimizable};
use crate::{ChatAdapter, Example, GLOBAL_SETTINGS, LM, Prediction, adapter::Adapter};

pub struct Predict {
    pub signature: Box<dyn MetaSignature>,
    pub tools: Vec<Arc<dyn ToolDyn>>,
}
```

### Implementation

```rust
impl Predict {
    pub fn new(signature: impl MetaSignature + 'static) -> Self {
        Self {
            signature: Box::new(signature),
            tools: vec![],
        }
    }

    pub fn new_with_tools(
        signature: impl MetaSignature + 'static,
        tools: Vec<Box<dyn ToolDyn>>,
    ) -> Self {
        Self {
            signature: Box::new(signature),
            tools: tools.into_iter().map(Arc::from).collect(),
        }
    }

    pub fn with_tools(mut self, tools: Vec<Box<dyn ToolDyn>>) -> Self {
        self.tools = tools.into_iter().map(Arc::from).collect();
        self
    }

    pub fn add_tool(mut self, tool: Box<dyn ToolDyn>) -> Self {
        self.tools.push(Arc::from(tool));
        self
    }
}

impl Predictor for Predict {
    async fn forward(&self, inputs: Example) -> anyhow::Result<Prediction> {
        let (adapter, lm) = {
            let guard = GLOBAL_SETTINGS.read().unwrap();
            let settings = guard.as_ref().unwrap();
            (settings.adapter.clone(), Arc::clone(&settings.lm))
        }; // guard is dropped here

        adapter
            .call(lm, self.signature.as_ref(), inputs, self.tools.clone())
            .await
    }

    async fn forward_with_config(
        &self,
        inputs: Example,
        lm: Arc<LM>,
    ) -> anyhow::Result<Prediction> {
        ChatAdapter
            .call(lm, self.signature.as_ref(), inputs, self.tools.clone())
            .await
    }
}

impl Optimizable for Predict {
    fn get_signature(&self) -> &dyn MetaSignature {
        self.signature.as_ref()
    }

    fn parameters(&mut self) -> IndexMap<String, &mut dyn Optimizable> {
        IndexMap::new()
    }

    fn update_signature_instruction(&mut self, instruction: String) -> anyhow::Result<()> {
        let _ = self.signature.update_instruction(instruction);
        Ok(())
    }
}
```

### Usage

```rust
use dspy_rs::{Predict, Signature, example};

#[derive(Signature)]
struct QuestionAnswer {
    #[input]
    question: String,

    #[output]
    answer: String,
}

// Create predictor
let predictor = Predict::new(QuestionAnswer::new());

// Forward inference
let input = example! {
    "question": "input" => "What is 2+2?"
};

let prediction = predictor.forward(input).await?;
let answer = prediction.get("answer", None);
```

---

## Module Trait

**Location**: `crates/dspy-rs/src/core/module.rs`

### Definition

```rust
use anyhow::Result;
use crate::{Example, Prediction};

#[allow(async_fn_in_trait)]
pub trait Module: Send + Sync {
    /// Forward pass
    async fn forward(&self, inputs: Example) -> Result<Prediction>;

    /// Batch processing with progress bar
    async fn batch(
        &self,
        inputs: Vec<Example>,
        max_concurrency: usize,
        display_progress: bool,
    ) -> Result<Vec<Prediction>> {
        let total = inputs.len();
        let mut pb = if display_progress {
            Some(tqdm!(total = total, desc = "Processing"))
        } else {
            None
        };

        // Pair each input with its index to maintain order
        let indexed_results: Vec<(usize, Result<Prediction>)> =
            stream::iter(inputs.into_iter().enumerate())
                .map(|(idx, example)| async move {
                    let result = self.forward(example).await;
                    (idx, result)
                })
                .buffer_unordered(max_concurrency)
                .inspect(|_| {
                    if let Some(ref mut progress) = pb {
                        let _ = progress.update(1);
                    }
                })
                .collect()
                .await;

        // Sort results back to original order
        let mut indexed_results = indexed_results;
        indexed_results.sort_by_key(|(idx, _)| *idx);

        // Collect predictions and handle errors
        let mut predictions = Vec::with_capacity(total);
        for (_, result) in indexed_results {
            predictions.push(result?);
        }

        Ok(predictions)
    }
}
```

**Note**: `Predictor` and `Module` are similar but separate traits. `Module` has a progress bar in batch processing.

---

## Optimizable Trait

**Location**: `crates/dspy-rs/src/core/module.rs`

### Definition

```rust
use indexmap::IndexMap;
use crate::core::MetaSignature;

#[allow(unused_variables)]
pub trait Optimizable {
    /// Get the signature for this module
    fn get_signature(&self) -> &dyn MetaSignature {
        todo!()
    }

    /// Get all optimizable sub-parameters (for nested modules)
    fn parameters(&mut self) -> IndexMap<String, &mut dyn Optimizable>;

    /// Update the signature instruction
    fn update_signature_instruction(&mut self, instruction: String) -> anyhow::Result<()> {
        todo!()
    }
}
```

**Purpose**: Allows optimizers to introspect and modify modules.

---

## Optimizer Trait

**Location**: `crates/dspy-rs/src/optimizer/mod.rs`

### Definition

```rust
use crate::{
    core::{Module, Optimizable},
    data::example::Example,
    evaluate::Evaluator,
};
use anyhow::Result;

#[allow(async_fn_in_trait)]
pub trait Optimizer {
    /// Compile/optimize a module using training data
    async fn compile<M>(&self, module: &mut M, trainset: Vec<Example>) -> Result<()>
    where
        M: Module + Optimizable + Evaluator;
}
```

### Built-in Optimizers

dspy-rs v0.7.3 includes three optimizers:

1. **MIPROv2** (`crates/dspy-rs/src/optimizer/mipro.rs`)
   - Multi-prompt Instruction Proposal Optimizer
   - Optimizes both instructions and demonstrations

2. **COPRO** (`crates/dspy-rs/src/optimizer/copro.rs`)
   - Coordinate Ascent Prompt Optimizer
   - Iteratively optimizes prompts using coordinate ascent

3. **GEPA** (`crates/dspy-rs/src/optimizer/gepa.rs`)
   - Genetic Prompt Algorithm
   - Uses genetic algorithm for prompt optimization

### Usage Example

```rust
use dspy_rs::{MIPROv2, Predict, QuestionAnswer, Evaluator};

// Create predictor
let mut predictor = Predict::new(QuestionAnswer::new());

// Create optimizer
let optimizer = MIPROv2::builder()
    .num_candidates(10)
    .build();

// Compile/optimize
optimizer.compile(&mut predictor, trainset).await?;
```

---

## Evaluator Trait

**Location**: `crates/dspy-rs/src/evaluate/evaluator.rs`

### Definition

```rust
use crate::core::Module;
use crate::data::{example::Example, prediction::Prediction};

#[allow(async_fn_in_trait)]
pub trait Evaluator: Module {
    const MAX_CONCURRENCY: usize = 32;
    const DISPLAY_PROGRESS: bool = true;

    /// Metric function - compute score for single prediction
    async fn metric(&self, example: &Example, prediction: &Prediction) -> f32;

    /// Evaluate on a dataset
    async fn evaluate(&self, examples: Vec<Example>) -> f32 {
        let predictions = self
            .batch(
                examples.clone(),
                Self::MAX_CONCURRENCY,
                Self::DISPLAY_PROGRESS,
            )
            .await
            .unwrap();

        let total = examples.len();

        // Pair examples with predictions and evaluate with controlled concurrency
        let metrics: Vec<f32> = stream::iter(examples.iter().zip(predictions.iter()).enumerate())
            .map(|(_, (example, prediction))| {
                let prediction = prediction.clone();
                async move { self.metric(example, &prediction).await }
            })
            .buffer_unordered(Self::MAX_CONCURRENCY)
            .collect()
            .await;

        metrics.iter().sum::<f32>() / total as f32
    }
}
```

### Usage Example

```rust
use dspy_rs::{Evaluator, Module, Predict, Example, Prediction};

impl Evaluator for Predict {
    async fn metric(&self, example: &Example, prediction: &Prediction) -> f32 {
        // Compare prediction to expected output
        let expected = example.get("answer", None).as_str().unwrap();
        let predicted = prediction.get("answer", None).as_str().unwrap();

        if expected == predicted {
            1.0
        } else {
            0.0
        }
    }
}

// Evaluate on test set
let score = predictor.evaluate(testset).await?;
println!("Accuracy: {}", score);
```

---

## Global Settings

**Location**: `crates/dspy-rs/src/core/settings.rs`

### Definition

```rust
use std::sync::{Arc, LazyLock, RwLock};
use super::LM;
use crate::adapter::Adapter;

pub struct Settings {
    pub lm: Arc<LM>,
    pub adapter: Arc<dyn Adapter>,
}

impl Settings {
    pub fn new(lm: LM, adapter: impl Adapter + 'static) -> Self {
        Self {
            lm: Arc::new(lm),
            adapter: Arc::new(adapter),
        }
    }
}

pub static GLOBAL_SETTINGS: LazyLock<RwLock<Option<Settings>>> =
    LazyLock::new(|| RwLock::new(None));

pub fn get_lm() -> Arc<LM> {
    Arc::clone(&GLOBAL_SETTINGS.read().unwrap().as_ref().unwrap().lm)
}

pub fn configure(lm: LM, adapter: impl Adapter + 'static) {
    let settings = Settings::new(lm, adapter);
    *GLOBAL_SETTINGS.write().unwrap() = Some(settings);
}
```

### Usage

```rust
use dspy_rs::{configure, ChatAdapter, LM};

// Configure global settings
let lm = LM::builder()
    .model("gpt-4o-mini")
    .api_key(std::env::var("OPENAI_API_KEY")?)
    .build()
    .await?;

configure(lm, ChatAdapter);

// Now all predictors will use these settings by default
let predictor = Predict::new(QuestionAnswer::new());
let result = predictor.forward(input).await?;
```

---

## DummyPredict (for Testing)

**Location**: `crates/dspy-rs/src/predictors/mod.rs`

```rust
pub struct DummyPredict;

impl Predictor for DummyPredict {
    async fn forward(&self, inputs: Example) -> anyhow::Result<Prediction> {
        Ok(Prediction::new(inputs.data, LmUsage::default()))
    }

    async fn forward_with_config(
        &self,
        inputs: Example,
        lm: Arc<LM>,
    ) -> anyhow::Result<Prediction> {
        Ok(Prediction::new(inputs.data, LmUsage::default()))
    }
}
```

**Use Case**: Testing without calling any actual LM.

---

## Summary

**Key Traits**:

1. ✅ **Predictor** - Forward inference interface (uses global settings)
2. ✅ **Module** - Similar to Predictor but with progress bar support
3. ✅ **Optimizable** - Allows optimizers to introspect/modify modules
4. ✅ **Optimizer** - Compiles/optimizes modules using training data
5. ✅ **Evaluator** - Evaluates module performance on datasets

**Built-in Implementations**:

1. ✅ **Predict** - Basic predictor using a signature
2. ✅ **DummyPredict** - Testing predictor (no LM calls)

**Built-in Optimizers**:

1. ✅ **MIPROv2** - Multi-prompt instruction proposal
2. ✅ **COPRO** - Coordinate ascent prompt optimization
3. ✅ **GEPA** - Genetic prompt algorithm

**Global Configuration**:

- ✅ `configure(lm, adapter)` - Set global LM and adapter
- ✅ `get_lm()` - Get global LM
- ✅ `GLOBAL_SETTINGS` - Global settings storage

**Do NOT**:
- ❌ Forget to call `configure()` before using predictors
- ❌ Assume Predictor and Module are the same (they're similar but separate)
- ❌ Try to optimize model weights (dspy-rs optimizes prompts/instructions)

**DO**:
- ✅ Implement `Evaluator` trait with a `metric()` function
- ✅ Use `batch()` for efficient parallel processing
- ✅ Use `Optimizable` trait if you want your module to be optimizable
- ✅ Use `DummyPredict` for testing

---

## Workflow Example

```rust
use dspy_rs::{configure, ChatAdapter, LM, Predict, Signature, MIPROv2, Evaluator, Module, example};

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

// 3. Create predictor
let mut predictor = Predict::new(QuestionAnswer::new());

// 4. Implement Evaluator
impl Evaluator for Predict {
    async fn metric(&self, example: &Example, prediction: &Prediction) -> f32 {
        // Your metric logic
        1.0
    }
}

// 5. Optimize
let optimizer = MIPROv2::builder().build();
optimizer.compile(&mut predictor, trainset).await?;

// 6. Evaluate
let score = predictor.evaluate(testset).await?;
println!("Score: {}", score);

// 7. Use
let input = example! { "question": "input" => "What is 2+2?" };
let result = predictor.forward(input).await?;
```

---

## References

- **Official Repo**: https://github.com/krypticmouse/DSRs
- **Predictor**: `crates/dspy-rs/src/predictors/mod.rs`
- **Predict**: `crates/dspy-rs/src/predictors/predict.rs`
- **Module**: `crates/dspy-rs/src/core/module.rs`
- **Optimizer**: `crates/dspy-rs/src/optimizer/mod.rs`
- **Evaluator**: `crates/dspy-rs/src/evaluate/evaluator.rs`
- **Settings**: `crates/dspy-rs/src/core/settings.rs`
- **Version**: 0.7.3 (verified 2025-11-16)
