# DSPy/DSRs Knowledge Base

This knowledge base provides indexed documentation for both DSPy (Python) and DSRs (Rust) to support the `dspy-researcher` agent.

## Directory Structure

```
.claude/knowledge/dspy/
├── source/             # ⭐ VERIFIED DSRs v0.7.3 source code documentation
│   ├── adapter-trait.md            # Adapter trait (THE trait to implement)
│   ├── lm-struct.md                # LM struct (NOT a trait)
│   ├── core-types.md               # Example, Prediction, Chat, Message, etc.
│   └── predictors-optimizers-evaluation.md  # Predictor, Optimizer, Evaluator
├── VERIFICATION.md     # ⚠️  Audit correction record (read this first!)
├── INDEX.md            # Complete navigational index for source code docs
├── learn/              # Core DSPy concepts with Rust translations
│   ├── 01-signatures.md
│   ├── 02-modules.md
│   ├── 03-language-models.md
│   ├── 04-optimizers.md
│   └── 05-metrics-evaluation.md
├── tutorials/          # Common patterns and use cases
│   └── rag-patterns.md
├── dsrs-examples/      # Annotated Rust examples from DSRs repo
│   ├── 01-simple-qa-pipeline.rs
│   ├── 02-miprov2-optimization.rs
│   ├── 03-module-iteration.rs
│   └── README.md
└── README.md          # This file
```

## Quick Reference

### Core Concepts

| Concept | Python (DSPy) | Rust (DSRs) | Doc |
|---------|---------------|-------------|-----|
| Signatures | `class QA(dspy.Signature)` | `#[Signature] struct QA` | [01-signatures.md](learn/01-signatures.md) |
| Modules | `class RAG(dspy.Module)` | `impl Module for RAG` | [02-modules.md](learn/02-modules.md) |
| Predictors | `dspy.Predict`, `dspy.ChainOfThought` | `Predict`, `ChainOfThought` | [02-modules.md](learn/02-modules.md) |
| LM Config | `dspy.configure(lm=...)` | `configure(LM::builder()...)` | [03-language-models.md](learn/03-language-models.md) |
| Optimization | `BootstrapFewShot`, `MIPROv2` | `MIPROv2`, `COPRO` | [04-optimizers.md](learn/04-optimizers.md) |
| Evaluation | `dspy.Evaluate` | `impl Evaluator` | [05-metrics-evaluation.md](learn/05-metrics-evaluation.md) |

### Common Patterns

- **RAG (Retrieval-Augmented Generation)**: [tutorials/rag-patterns.md](tutorials/rag-patterns.md)
- **Multi-Hop Search**: [tutorials/rag-patterns.md](tutorials/rag-patterns.md#multi-hop-rag)
- **Agent Patterns**: [tutorials/rag-patterns.md](tutorials/rag-patterns.md#rag-as-agent-pattern)

### Rust Examples

- **Simple QA Pipeline**: [dsrs-examples/01-simple-qa-pipeline.rs](dsrs-examples/01-simple-qa-pipeline.rs)
- **MIPROv2 Optimization**: [dsrs-examples/02-miprov2-optimization.rs](dsrs-examples/02-miprov2-optimization.rs)
- **Module Iteration**: [dsrs-examples/03-module-iteration.rs](dsrs-examples/03-module-iteration.rs)

## Using This Knowledge Base

### For the dspy-researcher Agent

The agent is configured to:
1. **Check local knowledge base FIRST** for fast, offline access
2. Search with `Grep` to find relevant patterns
3. Read specific files with `Read` for detailed information
4. Only use `WebFetch`/`WebSearch` for missing or updated information

### For Manual Reference

Use `grep` to find specific topics:
```bash
# Find all mentions of optimization
grep -r "optimization" .claude/knowledge/dspy/

# Find Rust examples of Signatures
grep -r "Signature" .claude/knowledge/dspy/dsrs-examples/

# Search for edge deployment patterns
grep -r "edge" .claude/knowledge/dspy/
```

## ⭐ NEW: Verified Source Code Documentation

**Added 2025-11-16**: Complete, verified dspy-rs v0.7.3 source code documentation

### Critical Files (READ THESE FIRST)

1. **[VERIFICATION.md](VERIFICATION.md)** - Why previous audits were wrong
   - ❌ What doesn't exist: LanguageModel trait, template parameter, kwargs parameter
   - ✅ What actually exists: Adapter trait, LM struct
   - How to verify yourself

2. **[INDEX.md](INDEX.md)** - Complete navigational index
   - Quick start guide
   - Implementation patterns
   - Common pitfalls
   - How to find what you need

3. **[source/adapter-trait.md](source/adapter-trait.md)** - THE trait to implement
   - Complete Adapter trait definition
   - ChatAdapter reference implementation
   - Usage examples for Candle

### All Source Code Documentation

- **[source/adapter-trait.md](source/adapter-trait.md)** - Adapter trait (format, parse_response, call)
- **[source/lm-struct.md](source/lm-struct.md)** - LM struct and how adapters use it
- **[source/core-types.md](source/core-types.md)** - Example, Prediction, Chat, Message, etc.
- **[source/predictors-optimizers-evaluation.md](source/predictors-optimizers-evaluation.md)** - Predictor, Optimizer, Evaluator

**Status**: ✅ Complete coverage of core dspy-rs v0.7.3 API

---

## Coverage

### What's Included
- ✅ **VERIFIED dspy-rs v0.7.3 source code** (NEW - extracted from official repo)
- ✅ Core DSPy concepts (Signatures, Modules, LMs, Optimizers, Metrics)
- ✅ Python → Rust translation examples
- ✅ RAG patterns with edge deployment considerations
- ✅ 3 fully annotated DSRs Rust examples
- ✅ References to all 11 DSRs examples

### What Requires Web Lookup
- Latest API changes (docs.rs)
- New optimizers or features added after v0.7.3
- Specific tutorial walkthroughs (dspy.ai)
- Community examples and patterns

## Updating This Knowledge Base

To add new content:
1. Add markdown files to appropriate directories
2. Update this README with new entries
3. Update the `dspy-researcher` agent if adding new categories

## External Resources

- **DSPy Learn**: https://dspy.ai/learn/
- **DSPy Tutorials**: https://dspy.ai/tutorials/
- **DSRs Documentation**: https://dsrs.herumbshandilya.com/
- **DSRs API Reference**: https://docs.rs/dspy-rs/latest/dspy_rs/
- **DSRs GitHub**: https://github.com/krypticmouse/DSRs
