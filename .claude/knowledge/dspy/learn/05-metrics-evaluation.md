# DSPy Metrics and Evaluation Documentation

---

# ⚠️ WARNING: PYTHON REFERENCE ONLY - THIS PROJECT USES RUST ⚠️

**This file contains Python DSPy examples for conceptual learning.**

**For Rust implementation, see:**
- [../source/predictors-optimizers-evaluation.md](../source/predictors-optimizers-evaluation.md) - Rust Evaluator trait and metric functions

**All code in this project MUST be Rust. Python examples below are for understanding concepts only.**

---

## What is a Metric?

A metric in DSPy is "a function that will take examples from your data and the output of your system and return a score that quantifies how good the output is." Metrics serve dual purposes: tracking progress during evaluation and enabling optimization of DSPy programs.

## Simple Metrics

Basic metrics are Python functions accepting three parameters:
- `example`: data from your training/dev set
- `pred`: output from your DSPy program
- `trace`: optional argument for optimization (can be ignored initially)

The function returns a numeric score (float, int, or bool).

**Simple Example:**
```python
def validate_answer(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()
```

Built-in utilities available:
- `dspy.evaluate.metrics.answer_exact_match`
- `dspy.evaluate.metrics.answer_passage_match`

## Evaluation Patterns

**Basic Loop:**
```python
scores = []
for x in devset:
    pred = program(**x.inputs())
    score = metric(x, pred)
    scores.append(score)
```

**Using the Evaluate Utility:**
```python
from dspy.evaluate import Evaluate
evaluator = Evaluate(devset=YOUR_DEVSET, num_threads=1,
                     display_progress=True, display_table=5)
evaluator(YOUR_PROGRAM, metric=YOUR_METRIC)
```

## AI-Powered Metrics

For long-form outputs, metrics should "check multiple dimensions of the output using AI feedback from LMs." This allows nuanced evaluation of complex system outputs.

## Key Principle

"Defining a good metric is an iterative process," requiring evaluation of actual outputs against your data to refine scoring approaches.

---

## Rust DSRs Equivalent

In dspy-rs, metrics implement the `Evaluator` trait:

```rust
use dspy_rs::evaluate::Evaluator;

struct CustomMetric;

impl Evaluator for CustomMetric {
    fn evaluate(&self, prediction: &Prediction, example: &Example) -> f64 {
        // Custom evaluation logic
        let pred_answer = prediction.get("answer", None);
        let expected = example.get("answer");

        if pred_answer.to_lowercase() == expected.to_lowercase() {
            1.0
        } else {
            0.0
        }
    }
}
```

**Key Features:**
- Type-safe evaluation
- Parallel evaluation using `rayon`
- Built-in metrics in `dspy_rs::evaluate` module
- Integration with optimizers
