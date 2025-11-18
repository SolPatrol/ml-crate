# DSPy Signatures: Complete Documentation

---

# ⚠️ WARNING: PYTHON REFERENCE ONLY - THIS PROJECT USES RUST ⚠️

**This file contains Python DSPy examples for conceptual learning.**

**For Rust implementation, see:**
- [../source/adapter-trait.md](../source/adapter-trait.md) - How to implement adapters in Rust
- [../source/core-types.md](../source/core-types.md) - Rust signature types (use `#[derive(Signature)]`)
- [../source/predictors-optimizers-evaluation.md](../source/predictors-optimizers-evaluation.md) - Rust predictors

**All code in this project MUST be Rust. Python examples below are for understanding concepts only.**

---

## Core Concept

DSPy Signatures are declarative specifications that define input/output behavior for language model modules. Rather than crafting prompts manually, signatures allow developers to specify *what* a task requires rather than *how* to prompt for it.

## Key Benefits

Signatures enable "modular and clean code, in which LM calls can be optimized into high-quality prompts (or automatic finetunes." They're more adaptive and reproducible than manual prompt engineering, and DSPy's compiler can generate optimized prompts that often exceed human-written versions.

## Two Syntax Approaches

### Inline Signatures (String-based)

Simple, concise syntax using arrow notation:
- `"question -> answer"` (defaults to `str` type)
- `"sentence -> sentiment: bool"`
- `"document -> summary"`
- `"context: list[str], question: str -> answer: str"`

Add instructions at runtime using the `instructions` keyword argument for dynamic customization.

### Class-based Signatures

More verbose, suited for complex tasks:

```python
class Emotion(dspy.Signature):
    """Classify emotion."""
    sentence: str = dspy.InputField()
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField()
```

Use `desc` parameters in `InputField` and `OutputField` to clarify task nature and specify output constraints.

## Type System

Signatures support:
- Basic types (`str`, `int`, `bool`)
- Typing module types (`list[str]`, `dict[str, int]`, `Optional[float]`)
- Custom Pydantic models
- Special data types (`dspy.Image`, `dspy.History`)
- Dot notation for nested types

## Best Practices

1. **Field naming matters semantically** — `question` differs from `answer`; `sql_query` differs from `python_code`
2. **Start simple** — avoid premature optimization; let DSPy compilers handle fine-tuning
3. **Compose signatures into modules** for building larger systems
4. **Compile modules** into optimized prompts or finetunes using DSPy optimizers

## Integration with DSPy Modules

Signatures work with modules like `dspy.Predict` and `dspy.ChainOfThought`, which may expand signatures (adding fields like `reasoning`) to enhance output quality.

---

## Rust DSRs Equivalent

In dspy-rs, signatures use Rust's type system with derive macros:

```rust
#[Signature]
struct Emotion {
    #[input]
    pub sentence: String,

    #[output]
    pub sentiment: String, // Could use enum for type safety
}
```

Key differences:
- Rust uses struct syntax vs Python classes
- Type annotations are compile-time enforced
- `#[input]` and `#[output]` attributes replace Python's `InputField()` and `OutputField()`
- Chain-of-thought available via `#[Signature(cot)]`
