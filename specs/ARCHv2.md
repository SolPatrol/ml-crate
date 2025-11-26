# DSPy-RS Standalone Library (v4)

**Version**: 0.4.0
**Last Updated**: 2025-11-26

---

## Overview

**ml-crate-dsrs** is a standalone Rust library for DSPy module management and inference execution.

**Core Responsibility**: Load DSPy modules, execute inference, return results.

```
ml-crate-dsrs/
└── src/
    ├── lib.rs                 # Public API
    ├── hardware/              # VRAM detection, backend selection
    ├── model_pool/            # Qwen model lifecycle (load, warmup, unload)
    ├── adapters/
    │   └── llamacpp/          # llama.cpp ↔ dspy-rs bridge
    ├── inference/             # DSPy module pool, execution
    └── tools/                 # Tool trait for Rhai integration
```

**4 core components. Inference execution. All GPU vendors supported.**

---

## Core Components

### 1. Hardware Manager
- Detects available VRAM
- Selects backend: Vulkan (default), CUDA, Metal, or CPU
- Runtime memory monitoring

### 2. Model Pool
- Loads GGUF model(s) into GPU memory via llama.cpp
- Warmup (first inference compiles kernels)
- Memory footprint: ~300MB on GPU (Q4_K_M quantized)
- Cold start: <2s

### 3. LlamaCpp Adapter
- Bridges dspy-rs ↔ llama.cpp for inference
- Implements dspy-rs `Adapter` trait
- Receives pre-loaded model from Model Pool
- Handles tokenization/detokenization (built into llama.cpp)
- Supports all GPU vendors via backend selection

### 4. DSPy Engine
- Loads pre-optimized DSPy modules into memory
- Module selection: <1ms (HashMap lookup)
- Executes inference (async internally with Tokio, provides sync wrapper)
- Provides Tool trait for Rhai scripting integration
- API: `load_module(name, module)` and `invoke(module_name, input)`

---

## Backend Support

```
llama.cpp (via llama-cpp-2 bindings)
    │
    ├── Vulkan  → AMD, NVIDIA, Intel GPUs (DEFAULT)
    ├── CUDA    → NVIDIA GPUs (+10-20% vs Vulkan)
    ├── Metal   → Apple Silicon
    └── CPU     → Fallback (AVX2/AVX512)
```

| Backend | NVIDIA | AMD | Intel | Apple | Default |
|---------|--------|-----|-------|-------|---------|
| Vulkan | ✅ | ✅ | ✅ | ❌ | ✅ **Yes** |
| CUDA | ✅ | ❌ | ❌ | ❌ | No |
| Metal | ❌ | ❌ | ❌ | ✅ | No |
| CPU | ✅ | ✅ | ✅ | ✅ | Fallback |

---

## Public API

```rust
pub use hardware::HardwareManager;
pub use model_pool::ModelPool;
pub use inference::DSPyEngine;
pub use adapters::llamacpp::LlamaCppAdapter;
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
│  Model Pool (Qwen GGUF)             │
│      ↓                              │
│  LlamaCpp Adapter                   │
│      ↓                              │
│  llama.cpp Inference ◄──────┐       │
│  (Vulkan/CUDA/Metal/CPU)    │       │
│      ↓                      │       │
│  Parse tool calls           │       │
│      ↓                      │       │
│  Execute tools (ToolRegistry)       │
│      ↓                      │       │
│  Tool results ──────────────┘       │
│  (may loop for multi-turn)          │
└─────────────────────────────────────┘
    ↓
User Application: Process result
```

---

## Model Format

| Attribute | Value |
|-----------|-------|
| Format | GGUF |
| Model | Qwen2.5-0.5B-Instruct |
| Quantization | Q4_K_M (recommended) |
| Size | ~300MB |
| Source | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) |

---

## Dependencies

- **`dspy-rs`** (v0.7.3+) - DSPy module system
- **`llama-cpp-2`** (v0.1+) - llama.cpp Rust bindings
- **`tokio`** (v1.0+) - Async runtime
- **`rhai`** (v1.0+) - Rhai scripting engine

### Feature Flags

```toml
[features]
default = ["vulkan"]  # Vulkan as default (broadest GPU support)

cuda = ["llama-cpp-2/cuda"]      # NVIDIA optimization
vulkan = ["llama-cpp-2/vulkan"]  # AMD, NVIDIA, Intel GPUs
metal = ["llama-cpp-2/metal"]    # Apple Silicon
cpu = []                          # Fallback (always available)
```

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

---

## Migration from v3 (Candle)

| Component | v3 (Candle) | v4 (llama.cpp) |
|-----------|-------------|----------------|
| Inference Engine | candle-core | llama-cpp-2 |
| Model Format | Safetensors | GGUF |
| Adapter | CandleAdapter | LlamaCppAdapter |
| GPU Support | NVIDIA, Apple | NVIDIA, AMD, Intel, Apple |
| Model Size | ~1GB (FP16) | ~300MB (Q4_K_M) |

See [03-multi-backend-strategy.md](03-multi-backend-strategy.md) for detailed migration plan.
