# DSPy-RS Model Crate

**Version**: 0.1.0
**Last Updated**: 2025-11-16

## Changelog

### v0.1.0 (2025-11-16)
- Initial architecture specification
- Migration from LoRA/smolagent-rs to dspy-rs approach
- Single model deployment (future: 2x models for parallel requests)
- DSPy Module-based agent specialization
- Internal `candle_adapter` module for dspy-rs ↔ Candle integration

---

## Overview

The DSPy-RS crate provides GPU-accelerated nano LLM inference for the Omakase Gaming project, using `dspy-rs` to enable dynamic agent behaviors through optimized prompt programming and tool integrations.

## Module Structure

```
ml-crate-dsrs/
└── src/
    ├── lib.rs                 # Main entry point
    ├── hardware/              # Component #1: VRAM detection
    ├── model_pool/            # Component #2: Qwen3 lifecycle
    ├── candle_adapter/        # Interface layer (implements dspy-rs LM trait)
    │   ├── mod.rs             # Re-exports
    │   ├── model.rs           # CandleLanguageModel (implements dspy_rs::lm::LanguageModel)
    │   └── config.rs          # AdapterConfig (internal configuration)
    ├── agent/                 # Component #3: DSPy Module registry
    ├── context/               # Component #4: Game state formatting
    ├── tools/                 # Component #5: Tool execution
    └── inference/             # Component #6: High-level API
```

**Note**: The `candle_adapter` module is an internal **interface layer** that bridges dspy-rs with Candle. It **implements** the dspy-rs `LanguageModel` trait (doesn't define it) and provides Candle-based inference that components #3 and #6 depend on.

## Core Components

### 1. Hardware Manager

**Responsibility**: VRAM detection and execution backend selection.

- Queries available VRAM at initialization
- Exposes `BackendConfig` enum: `GPU(device_id)` | `CPU(threads)`
- Provides runtime memory monitoring for graceful degradation

### 2. Model Pool

**Responsibility**: Qwen3-0.6B instance management.

- Maintains 1 pre-loaded model instance in memory
- Handles model lifecycle: initialization, warmup, unloading
- Memory footprint target: ~1GB on GPU
- **Future Enhancement**: Scale to 2 concurrent instances for parallel request handling

### 3. Agent Registry

**Responsibility**: Dynamic agent behavior via DSPy Module selection.

- Stores optimized DSPy Module instances per agent type
- Instant Module selection (<1ms) based on agent_type
- Each Module = Signature + Predictor (ReAct/ChainOfThought/Predict) + Optimized State
- Supports Module hot-reload for A/B testing

### 4. Context Builder

**Responsibility**: Game state formatting per agent type.

- Formats relevant game state data based on agent_type
- Dynamic context injection per request
- Separates context logic from agent logic

### 5. Tool Registry

**Responsibility**: Agentic capability exposure via function calling.

```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    async fn execute(&self, args: serde_json::Value) -> Result<ToolOutput>;
}
```

- **RAG Integration**: Vector search interface (embeddings + similarity lookup)
- **Game Server Tools**: Direct FFI or RPC bindings to game engine functions
- Tool discovery and schema validation at registration

### 6. Inference API

**Responsibility**: High-level generation interface.

```rust
pub struct InferenceRequest {
    pub prompt: String,
    pub agent_type: String,  // Selects DSPy Module
    pub game_state: GameState,
    pub max_tokens: u32,
}

pub async fn generate(req: InferenceRequest) -> Result<AgentResponse>;
```

- Selects Module from Agent Registry
- Builds context via Context Builder
- Executes Module.forward() for generation + tool calls
- Thread-safe request queuing

## Data Flow

```
Request → Model Pool (acquire model, queue if busy)
        → Agent Registry (select Module instance)
        → Context Builder (format game state)
        → dspy-rs Module.forward() (ReAct/ChainOfThought execution + tool calls)
        → Tool Registry (execute tools)
        → Response (release model)
```

## Performance Targets

- Cold start: <2s (model load)
- Module selection: <1ms (vs 100ms LoRA swap)
- Token generation: 30-50 tokens/sec on GPU
- Concurrent requests: Serialized with request queuing (future: parallel with 2x models)

## Optimization

**DSPy Module Optimization** (offline process):

- **COPRO** (Coordinate Ascent): Fast iterative prompt refinement
- **MIPROv2** (Multi-prompt Proposal): Comprehensive candidate evaluation
- Both optimize Module instructions + few-shot demonstrations
- Model weights remain frozen

**Module Serialization**: Save/load optimized Module state (prompts + demos) for deployment.

## Dependencies

### Core Framework Dependencies

These dependencies are used **across the entire crate**:

- **`dspy-rs`** (v0.7.3+)
  - Agent Module system (Signatures, Predictors, Optimizers)
  - ReAct/ChainOfThought/Predict patterns
  - COPRO/MIPROv2 optimization
  - Integrated via internal `candle_adapter` module

- **`tokio`** (v1.0+, features: `["full"]`)
  - Async runtime for non-blocking inference
  - Used by all async modules (inference, tools, agent)

- **`serde`**, **`serde_json`** (v1.0+)
  - Tool schema definition and Module serialization
  - Used across all modules for configuration and data exchange

- **`thiserror`** (v1.0+)
  - Error type definitions across all modules
  - Each module defines its own error types using thiserror

- **`tracing`** (v0.1+)
  - Structured logging for debugging and performance monitoring
  - All modules emit logs via tracing

### Model Inference Dependencies

These dependencies are **primarily used by `model_pool` and `candle_adapter`**:

- **`candle-core`**, **`candle-transformers`**, **`candle-nn`** (v0.8+)
  - GPU-accelerated model inference (Qwen3-0.6B)
  - Direct integration via `candle_adapter::CandleLanguageModel`
  - Adapts Candle's Qwen examples for dspy-rs compatibility

- **`tokenizers`** (v0.21+)
  - Qwen3 tokenization/detokenization
  - Shared via Arc between model_pool and candle_adapter

- **`anyhow`** (v1.0+)
  - Convenience error handling in model initialization and inference pipelines

### Internal Module Architecture

The `candle_adapter` module implements dspy-rs's `LanguageModel` trait:
- **Location**: `src/candle_adapter/`
- **Purpose**: Implements dspy-rs `LanguageModel` trait using Candle for inference
- **Key Components**:
  - `mod.rs` or `model.rs`: `CandleLanguageModel` struct (implements `dspy_rs::lm::LanguageModel`)
  - Uses dspy-rs types: `GenerateOptions`, `GenerateOutput`, `LMError`, `Message`, `TokenUsage`
  - Returns raw model output - dspy-rs Modules (ReAct/ChainOfThought) handle tool parsing
- **Specification**: See [specs/01-candle-adapter.md](specs/01-candle-adapter.md) (v0.2.0)

---