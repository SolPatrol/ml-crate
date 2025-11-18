# DSPy RAG (Retrieval-Augmented Generation) Patterns

---

# ⚠️ WARNING: PYTHON REFERENCE ONLY - THIS PROJECT USES RUST ⚠️

**This file contains Python DSPy tutorial examples for conceptual learning.**

**For Rust implementation, see:**
- [../source/adapter-trait.md](../source/adapter-trait.md) - How to implement custom RAG adapters in Rust
- [../source/predictors-optimizers-evaluation.md](../source/predictors-optimizers-evaluation.md) - Rust predictors and modules
- [../dsrs-examples/README.md](../dsrs-examples/README.md) - Rust example implementations

**All code in this project MUST be Rust. Python examples below are for understanding RAG patterns only.**

---

## Overview

RAG combines information retrieval with language models for enhanced question-answering. DSPy provides structured approaches to building, evaluating, and optimizing RAG systems.

## Key Tutorial Resources

- **Basic RAG**: https://dspy.ai/tutorials/rag/
- **Multi-Hop RAG**: https://dspy.ai/tutorials/multihop_search/
- **RAG as Agent**: https://dspy.ai/tutorials/agents/

## Core RAG Pattern

### Basic Implementation

1. **Configure Environment**: Set up LM and retriever
2. **Build Modules**: Create RAG module combining retrieval + generation
3. **Evaluate**: Define metrics and test on dataset
4. **Optimize**: Use DSPy optimizers to improve prompts

### RAG Module Structure

```python
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate_answer(context=context, question=question)
```

## Multi-Hop RAG

For complex questions requiring multiple retrieval steps:

### Architecture

- Multiple retrieval rounds
- Intermediate reasoning steps
- Query reformulation based on retrieved context

### Pattern

```python
class MultiHopRAG(dspy.Module):
    def __init__(self):
        self.generate_query = dspy.ChainOfThought("context, question -> search_query")
        self.retrieve = dspy.Retrieve(k=3)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # First hop
        query1 = self.generate_query(context="", question=question)
        context1 = self.retrieve(query1.search_query).passages

        # Second hop
        query2 = self.generate_query(context=context1, question=question)
        context2 = self.retrieve(query2.search_query).passages

        # Final answer
        all_context = context1 + context2
        return self.generate_answer(context=all_context, question=question)
```

## RAG as Agent Pattern

Using ReAct architecture for dynamic retrieval:

### Key Features

- Agent decides when to retrieve
- Tool-based retrieval integration
- Dynamic query formulation

### Pattern

```python
class RAGAgent(dspy.Module):
    def __init__(self):
        self.agent = dspy.ReAct(
            signature="question -> answer",
            tools=[retrieve_tool, search_tool]
        )

    def forward(self, question):
        return self.agent(question=question)
```

## Optimization Strategies

### For Basic RAG

- Use `BootstrapFewShot` with ~10 examples
- Optimize both retrieval query generation and answer generation

### For Multi-Hop RAG

- Use `MIPROv2` with larger datasets (50+ examples)
- Optimize query reformulation at each hop
- Tune number of hops based on complexity

### Evaluation Metrics

```python
def rag_metric(example, pred, trace=None):
    # Check answer correctness
    answer_match = example.answer.lower() in pred.answer.lower()

    # Optional: Check retrieval quality
    # Has relevant context been retrieved?

    return float(answer_match)
```

## Best Practices

1. **Start Simple**: Begin with basic RAG, add complexity as needed
2. **Evaluate Early**: Define metrics before optimization
3. **Iterate on Retrieval**: Query quality often matters more than model size
4. **Monitor Costs**: Track token usage during development
5. **Save Optimized Programs**: Persist optimized prompts for production

## Common Patterns for Edge Deployment

- **Pre-compute embeddings**: Reduce runtime retrieval costs
- **Optimize for latency**: Use faster models for query generation
- **Cache frequent queries**: Store common question-answer pairs
- **Minimize hops**: Each hop adds latency; optimize hop count

---

## Rust DSRs Considerations

For edge RAG with dspy-rs:

```rust
struct EdgeRAG {
    retrieve: RetrievalFunction,
    generate: Predict<QASignature>,
}

#[async_trait]
impl Module for EdgeRAG {
    async fn forward(&self, input: Example) -> Result<Prediction> {
        let question = input.get("question");

        // Async retrieval (could be local vector DB)
        let context = (self.retrieve)(question).await?;

        // Generate with nano LLM
        let example = example! {
            "context": "input" => context,
            "question": "input" => question,
        };

        self.generate.forward(example).await
    }
}
```

**Key considerations**:
- Local vector stores (Qdrant, LanceDB)
- Async I/O for retrieval
- Nano LLM for generation (edge-hosted)
- Minimal memory footprint
