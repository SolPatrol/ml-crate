# DSPy Engine - Planning Checklist

**Status**: 📋 Planning Complete
**Date**: 2025-11-24

---

## Key Design Decisions

### 1. Tool Calling Strategy
- **Approach**: Structured output + Rust ToolWrapper
- **Why**: Model-agnostic (works with any model in Model Pool)
- **How**: DSPy signature defines `tool_call` as optional output field, ToolWrapper parses and executes

### 2. Predictor Types
- **Available now**: `Predict`, `ChainOfThought`
- **Deferred**: `ReAct` (not implemented in dspy-rs v0.7.3)
- **Note**: We implement our own tool loop via ToolWrapper instead of waiting for ReAct

### 3. Tool Loop Location
- **Decision**: Rust-side (ToolWrapper), not Rhai
- **Why**: Performance, reusability across predictor types, type safety, single place to maintain

### 4. Rhai Integration
- **Role**: Rhai is THE interface (all invocations come from Rhai scripts)
- **API**: `invoke()` (no tools), `invoke_with_tools()` (tool-enabled)
- **Tools**: Rhai functions wrapped as Tool trait via RhaiTool

### 5. Module Storage
- **Format**: JSON (human-readable for dev iteration)
- **Structure**: Nested directories + manifest.json for fast lookup
- **Hot Reload**: All modules (file watcher on modules/ directory)

### 6. Async Strategy
- **Internal**: Tokio async
- **Rhai interface**: Sync wrappers (`invoke_sync`, `invoke_with_tools_sync`) using `block_on`

---

## Research Findings

### dspy-rs v0.7.3 Limitations
- ❌ No ReAct predictor
- ❌ No Tool trait or tool support
- ❌ No built-in module serialization
- ✅ Has Predict, ChainOfThought
- ✅ Has Adapter trait (implemented by CandleAdapter)

### Tool Calling Solution
- Use DSPy signature to force structured output with `tool_call` field
- DSPy optimization trains model to output correct format
- Model-agnostic (not Qwen-specific parsing)

---

## Implementation Checklist

### Phase 3A: Core Engine
- [ ] `OptimizedModule` struct and JSON deserialization
- [ ] `SignatureDefinition` struct
- [ ] `ModuleManifest` struct and loading
- [ ] `DSPyEngine::new()` - loads manifest and modules
- [ ] `DSPyEngine::invoke()` - basic invocation
- [ ] `DSPyEngine::get_module()` - module lookup
- [ ] `execute_predict()` - Predict predictor execution
- [ ] `execute_chain_of_thought()` - CoT predictor execution
- [ ] Prompt building from module (instruction + demos + input)
- [ ] Output parsing from LLM response
- [ ] Hot reload via file watcher
- [ ] Unit tests for module loading

### Phase 3B: Tool System
- [ ] `Tool` trait definition
- [ ] `ToolError` enum
- [ ] `ToolRegistry` - stores tools, executes by name
- [ ] `ToolRegistry::to_json()` - generates available_tools for LLM
- [ ] `ToolCall` struct - parsed tool request
- [ ] `ToolWrapper` - generic wrapper for any predictor
- [ ] `ToolWrapper::invoke()` - tool loop implementation
- [ ] Max iterations limit
- [ ] Context injection (tool results into input)
- [ ] Unit tests for tool execution
- [ ] Unit tests for tool loop

### Phase 3C: Rhai Integration
- [ ] `RhaiTool` - wraps Rhai function as Tool
- [ ] `json_to_dynamic()` / `dynamic_to_json()` helpers
- [ ] `register_dspy_engine()` - registers engine in Rhai
- [ ] `invoke_sync()` - sync wrapper for Rhai
- [ ] `invoke_with_tools_sync()` - sync wrapper with tools
- [ ] `register_tool()` - Rhai function to register tools
- [ ] End-to-end Rhai test (simple invoke)
- [ ] End-to-end Rhai test (tool invoke)

### Phase 3D: Integration
- [ ] Integration with CandleAdapter
- [ ] Integration with Model Pool
- [ ] Example modules in modules/ directory
- [ ] Example Rhai scripts
- [ ] Documentation

---

## File Structure

```
src/
├── inference/
│   ├── mod.rs              # pub use exports
│   ├── engine.rs           # DSPyEngine
│   ├── module.rs           # OptimizedModule, SignatureDefinition
│   ├── manifest.rs         # ModuleManifest
│   └── error.rs            # DSPyEngineError
├── tools/
│   ├── mod.rs              # pub use exports
│   ├── traits.rs           # Tool trait
│   ├── registry.rs         # ToolRegistry
│   ├── wrapper.rs          # ToolWrapper
│   └── rhai_tool.rs        # RhaiTool
└── rhai/
    ├── mod.rs              # pub use exports
    └── registration.rs     # register_dspy_engine()

modules/
├── manifest.json
├── npc/
│   └── dialogue_casual.json
└── test/
    └── tool_test.json
```

---

## Dependencies to Add

```toml
# In Cargo.toml
rhai = "1.0"          # Rhai scripting engine
notify = "6.0"        # File watching for hot reload
```

---

## Open Questions

1. **Prompt format**: How exactly should we format the prompt from module + input? Need to verify against dspy-rs.

2. **Output parsing**: What parsing strategy for structured output? (field markers, JSON, etc.)

3. **Error recovery**: What happens if tool execution fails mid-loop?

4. **Tool schema validation**: Should we validate tool args against schema before execution?

5. **Concurrent tool calls**: Should we support parallel tool execution in the future?

---

## References

- [ARCH.md](../ARCH.md) - System architecture
- [specs/01-candle-adapter.md](./01-candle-adapter.md) - CandleAdapter (complete)
- [specs/08-dspy-engine.md](./08-dspy-engine.md) - Full specification
- `.claude/knowledge/dspy/source/` - Verified dspy-rs source
