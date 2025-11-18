# DSPy Optimizers: Complete Documentation

---

# ⚠️ WARNING: PYTHON REFERENCE ONLY - THIS PROJECT USES RUST ⚠️

**This file contains Python DSPy examples for conceptual learning.**

**For Rust implementation, see:**
- [../source/predictors-optimizers-evaluation.md](../source/predictors-optimizers-evaluation.md) - Rust Optimizer trait (MIPROv2, COPRO, GEPA)

**All code in this project MUST be Rust. Python examples below are for understanding concepts only.**

---

## Overview

DSPy optimizers are algorithms that tune program parameters—prompts and/or LM weights—to maximize specified metrics like accuracy. They require three inputs: your DSPy program, a metric function, and training data (which can be minimal).

## Available Optimizer Categories

### Automatic Few-Shot Learning

These optimizers generate optimized examples within prompts:

1. **LabeledFewShot**: "Simply constructs few-shot examples (demos) from provided labeled input and output data points." Requires specifying `k` examples and a trainset.

2. **BootstrapFewShot**: Uses a teacher module to generate demonstrations for each program stage. Parameters include `max_labeled_demos` and `max_bootstrapped_demos`. The metric validates which demonstrations are included.

3. **BootstrapFewShotWithRandomSearch**: Applies BootstrapFewShot multiple times with random search, selecting the best program. Adds `num_candidate_programs` parameter.

4. **KNNFewShot**: Uses k-Nearest Neighbors to find training examples most similar to inputs, then applies BootstrapFewShot optimization.

### Automatic Instruction Optimization

These optimizers refine natural language instructions:

1. **COPRO**: "Generates and refines new instructions for each step, and optimizes them with coordinate ascent." Uses `depth` parameter for iteration count.

2. **MIPROv2**: "Generates instructions _and_ few-shot examples in each step." Data-aware and demonstration-aware, employing Bayesian Optimization.

3. **SIMBA**: Uses stochastic sampling to identify challenging examples, then applies LLM introspection to generate improvement rules.

4. **GEPA**: Leverages LM reflection on program trajectories to identify gaps and propose improvements. Supports domain-specific feedback.

### Automatic Finetuning

**BootstrapFinetune**: "Distills a prompt-based DSPy program into weight updates," producing a program with finetuned models replacing prompted LMs.

### Program Transformations

**Ensemble**: Combines multiple DSPy programs, using the full set or randomly sampling subsets.

## Selection Guidance

- **~10 examples**: Start with BootstrapFewShot
- **50+ examples**: Try BootstrapFewShotWithRandomSearch
- **0-shot optimization only**: Use MIPROv2 configured for 0-shot
- **40+ trials, 200+ examples**: Try MIPROv2 for longer runs
- **Efficiency priority**: Finetune small LM with BootstrapFinetune

## Usage Pattern

All optimizers share a consistent interface:

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

config = dict(max_bootstrapped_demos=4, max_labeled_demos=4,
              num_candidate_programs=10, num_threads=4)
optimizer = BootstrapFewShotWithRandomSearch(metric=YOUR_METRIC, **config)
optimized_program = optimizer.compile(YOUR_PROGRAM, trainset=YOUR_TRAINSET)
```

## Cost Considerations

Typical optimization runs cost approximately $2 USD and take around ten minutes, though costs range from cents to tens of dollars depending on LM size, dataset size, and configuration.

## Saving and Loading

Programs can be persisted as JSON:

```python
optimized_program.save(path)
loaded_program = YOUR_CLASS()
loaded_program.load(path=path)
```

---

## Rust DSRs Equivalent

DSRs supports COPRO and MIPROv2 optimizers:

```rust
use dspy_rs::optimizer::{COPRO, MIPROv2};

// COPRO optimization
let optimizer = COPRO::new(metric, depth);
let optimized = optimizer.compile(program, trainset).await?;

// MIPROv2 optimization
let optimizer = MIPROv2::new(metric, config);
let optimized = optimizer.compile(program, trainset).await?;
```

**Key Differences:**
- Async optimization in Rust
- Type-safe metric functions
- Builder pattern for configuration
- Rayon for parallel processing

**For Edge Deployment:**
Optimization typically happens during development, not at runtime. The optimized prompts/instructions are then embedded in the deployed edge application.
