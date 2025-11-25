# DSPy-RS Standalone Library (v3)

**Version**: 0.3.0
**Last Updated**: 2025-11-20

---

## Overview

**ml-crate-dsrs** is a standalone Rust library for DSPy module management and inference execution.

**Core Responsibility**: Load DSPy modules, execute inference, return results.

```
ml-crate-dsrs/
└── src/
    ├── lib.rs                 # Public API
    ├── hardware/              # VRAM detection, backend selection
    ├── model_pool/            # Qwen3 lifecycle (load, warmup, unload)
    ├── candle_adapter/        # Candle ↔ dspy-rs bridge
    ├── inference/             # DSPy module pool, execution
    └── tools/                 # Tool trait for Rhai integration
```

**4 core components. Inference execution.**

---

## Core Components

### 1. Hardware Manager
- Detects available VRAM
- Selects backend: `GPU(device_id)`
- Runtime memory monitoring

### 2. Model Pool
- Loads model(s) into GPU memory
- Warmup (first inference compiles kernels)
- Memory footprint: ~1GB on GPU
- Cold start: <2s

### 3. Candle Adapter
- Bridges dspy-rs ↔ Candle for inference
- Implements dspy-rs `Adapter` trait
- Receives pre-loaded model from Model Pool
- Handles tokenization/detokenization
- Status: ✅ Complete (v1.0.0)

### 4. DSPy Engine
- Loads pre-optimized DSPy modules into memory
- Module selection: <1ms (HashMap lookup)
- Executes inference (async internally with Tokio, provides sync wrapper)
- Provides Tool trait for Rhai scripting integration
- API: `load_module(name, module)` and `invoke(module_name, input)`

---

## Public API

```rust
pub use hardware::HardwareManager;
pub use model_pool::ModelPool;
pub use inference::DSPyEngine;
pub use tools::{Tool, ToolRegistry};  // For Rhai scripting integration
```

---

## Data Flow

```
User Application
    ↓
engine.invoke("module_name", input)
    ↓
┌─────────────────────────────────────┐
│ ml-crate-dsrs                       │
│                                     │
│  DSPy Engine                        │
│      ↓                              │
│  Module Selection (HashMap)         │
│      ↓                              │
│  Module.forward(input, model_pool)  │
│      ↓                              │
│  Model Pool (Qwen3)                 │
│      ↓                              │
│  Candle Adapter                     │
│      ↓                              │
│  GPU Inference ◄─────────┐          │
│      ↓                   │          │
│  Parse tool calls        │          │
│      ↓                   │          │
│  Execute tools (ToolRegistry)       │
│      ↓                   │          │
│  Tool results ───────────┘          │
│  (may loop for multi-turn)          │
└─────────────────────────────────────┘
    ↓
User Application: Process result
```

---

## Dependencies

- **`dspy-rs`** (v0.7.3+) - DSPy module system
- **`candle-core`** (v0.8+) - GPU inference
- **`candle-transformers`** (v0.8+) - Qwen2.5 model
- **`tokenizers`** (v0.21+) - Tokenization
- **`tokio`** (v1.0+) - Async runtime
- **`rhai`** (v1.0+) - Rhai scripting engine

---

## Rhai Integration

**Rhai is required** - ml-crate-dsrs uses Rhai for extensibility:

- Game server provides Rhai instance to ml-crate-dsrs
- ml-crate-dsrs loads scripts from modules/ directory
- Rhai scripts implement Tool trait (RAG, state queries, mutators)
- Tools registered with ToolRegistry
- Scripts call `llm_manager.invoke(module, input)` for inference
- DSPy modules call tools during inference via custom parsing layer

**Tool trait is generic** - works for any function type (RAG, APIs, mutators, file operations)