# DSPy Engine Specification

**Version**: 1.0.0
**Status**: ✅ Phase 3A Complete
**Dependencies**: CandleAdapter (Component #3), Model Pool (Component #2), dspy-rs (v0.7.3+)
**Last Updated**: 2025-11-24

---

## ✅ Implementation Status (v1.0.0)

**Phase 3A (Core Engine)**: ✅ COMPLETE
- OptimizedModule, Demo, SignatureDefinition types implemented
- ModuleManifest with load/reload functionality
- SignatureRegistry with generic registration pattern
- Value ↔ Example/Prediction conversion helpers
- DSPyEngine orchestrator with invoke() for Predict and ChainOfThought
- Full local inference via CandleAdapter (no external API calls)
- 73 unit tests passing (63 run + 10 ignored requiring model)
- 16 integration tests passing (10 run + 6 ignored requiring model)
- All 6 model-dependent tests verified passing with real Qwen2.5-0.5B

**Files Created**:
- `src/inference/mod.rs` - Module exports
- `src/inference/error.rs` - DSPyEngineError enum (12 variants)
- `src/inference/module.rs` - OptimizedModule, Demo, PredictorType, etc.
- `src/inference/manifest.rs` - ModuleManifest, ModuleEntry, load helpers
- `src/inference/registry.rs` - SignatureRegistry
- `src/inference/conversion.rs` - Value/Example conversion helpers
- `src/inference/engine.rs` - DSPyEngine struct
- `tests/fixtures/modules/*.json` - Test module fixtures
- `tests/dspy_engine_tests.rs` - Integration tests

**Next Phases**:
- Phase 3A-Optional: Hot Reload (file watcher)
- Phase 3B: Tool System (ToolRegistry, ToolWrapper)
- Phase 3C: Rhai Integration

---

## Overview

The DSPy Engine is the core orchestrator for loading pre-optimized DSPy modules, executing inference, and handling tool calls. It provides a Rhai-friendly API for game server integration.

**Module Path**: `ml_crate_dsrs::inference`
**Related Components**: CandleAdapter, Model Pool, Rhai scripting

---

## Architecture

```
Rhai Script (game server)
    ↓
llm_manager.invoke("module", input)
    or
llm_manager.invoke_with_tools("module", input)
    ↓
┌─────────────────────────────────────────────────────────┐
│ DSPyEngine                                              │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ ModuleRegistry                                   │   │
│  │   HashMap<String, OptimizedModule>               │   │
│  │   - Loaded from modules/ directory               │   │
│  │   - Hot reload on file change                    │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │ SignatureRegistry                                │   │
│  │   - Consumer registers signatures at startup     │   │
│  │   - Maps signature_name → MetaSignature factory  │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │ ToolWrapper (for tool-enabled modules only)      │   │
│  │   - Parses structured output for tool_call       │   │
│  │   - Executes tools via ToolRegistry              │   │
│  │   - Loops until final response (max iterations)  │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Predictor (Predict | ChainOfThought)             │   │
│  │   - Built from OptimizedModule + SignatureRegistry│   │
│  │   - Uses dspy-rs global settings (configure())   │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │ CandleAdapter → Model Pool → GPU Inference       │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
    ↓
Rhai Script: receives result
```

---

## Core Components

### 1. OptimizedModule

Represents a pre-optimized DSPy module loaded from disk.

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Predictor execution strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictorType {
    Predict,
    ChainOfThought,
    // ReAct - TODO: Phase 4 (requires upstream dspy-rs support)
}

/// Few-shot demonstration example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Demo {
    pub inputs: HashMap<String, serde_json::Value>,
    pub outputs: HashMap<String, serde_json::Value>,
}

/// Pre-optimized module loaded from JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedModule {
    /// Unique module identifier (e.g., "npc.dialogue.casual")
    pub module_id: String,

    /// Predictor type determines execution behavior
    pub predictor_type: PredictorType,

    /// Signature name - must match a registered signature in SignatureRegistry
    /// The consumer registers signatures at startup; this field references them by name
    pub signature_name: String,

    /// Signature definition (input/output fields) - for documentation/validation only
    /// The actual signature comes from SignatureRegistry, but this provides metadata
    #[serde(default)]
    pub signature: Option<SignatureDefinition>,

    /// Optimized instruction (from MIPROv2/COPRO)
    pub instruction: String,

    /// Few-shot demonstrations (from BootstrapFewShot/MIPROv2)
    pub demos: Vec<Demo>,

    /// Whether this module can request tool calls
    pub tool_enabled: bool,

    /// Metadata for debugging and versioning
    pub metadata: ModuleMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureDefinition {
    /// Input field names and descriptions
    pub inputs: Vec<FieldDefinition>,
    /// Output field names and descriptions
    pub outputs: Vec<FieldDefinition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    pub name: String,
    pub description: Option<String>,
    pub field_type: String,  // "string", "json", etc.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleMetadata {
    pub optimizer: String,           // "MIPROv2", "COPRO", "manual"
    pub optimized_at: Option<String>, // ISO 8601 timestamp
    pub metric_score: Option<f32>,   // Optimization metric score
    pub version: String,
}
```

### 2. Module File Format (JSON)

```json
{
  "module_id": "npc.dialogue.casual",
  "predictor_type": "predict",
  "signature_name": "npc.dialogue",
  "signature": {
    "inputs": [
      { "name": "npc_personality", "description": "NPC's personality traits", "field_type": "string" },
      { "name": "player_message", "description": "What the player said", "field_type": "string" },
      { "name": "conversation_history", "description": "Previous exchanges", "field_type": "string" }
    ],
    "outputs": [
      { "name": "response", "description": "NPC's response to the player", "field_type": "string" },
      { "name": "emotion", "description": "NPC's emotional state", "field_type": "string" }
    ]
  },
  "instruction": "You are roleplaying as an NPC in a fantasy game. Respond in character based on the personality provided. Keep responses concise (1-3 sentences) and natural.",
  "demos": [
    {
      "inputs": {
        "npc_personality": "Gruff blacksmith, distrustful of strangers",
        "player_message": "Hello there!",
        "conversation_history": ""
      },
      "outputs": {
        "response": "*grunts* What do you want? I'm busy.",
        "emotion": "annoyed"
      }
    }
  ],
  "tool_enabled": false,
  "metadata": {
    "optimizer": "MIPROv2",
    "optimized_at": "2025-11-24T10:30:00Z",
    "metric_score": 0.87,
    "version": "1.0.0"
  }
}
```

**Note:** The `signature_name` field references a signature registered in the consumer's `SignatureRegistry`.
The `signature` field is optional metadata for documentation/tooling - the actual signature type comes from the registry.

### 3. Tool-Enabled Module Format

For modules that can request tool calls:

```json
{
  "module_id": "npc.merchant.haggle",
  "predictor_type": "predict",
  "signature_name": "npc.merchant.haggle",
  "signature": {
    "inputs": [
      { "name": "query", "description": "Player's request", "field_type": "string" },
      { "name": "context", "description": "Current context including tool results", "field_type": "string" },
      { "name": "available_tools", "description": "JSON list of available tools", "field_type": "json" }
    ],
    "outputs": [
      { "name": "response", "description": "Response to player (if no tool needed)", "field_type": "string" },
      { "name": "tool_call", "description": "Tool to call (null if not needed)", "field_type": "json" }
    ]
  },
  "instruction": "You are a merchant NPC. If you need information (inventory, prices, player gold), request it via tool_call. Otherwise, respond directly.",
  "demos": [
    {
      "inputs": {
        "query": "How much gold do I have?",
        "context": "",
        "available_tools": "[{\"name\": \"get_player_gold\", \"description\": \"Get player's current gold\"}]"
      },
      "outputs": {
        "response": "",
        "tool_call": { "name": "get_player_gold", "args": {} }
      }
    },
    {
      "inputs": {
        "query": "How much gold do I have?",
        "context": "Tool result: Player has 500 gold",
        "available_tools": "[{\"name\": \"get_player_gold\", \"description\": \"Get player's current gold\"}]"
      },
      "outputs": {
        "response": "You have 500 gold coins, traveler.",
        "tool_call": null
      }
    }
  ],
  "tool_enabled": true,
  "metadata": {
    "optimizer": "MIPROv2",
    "optimized_at": "2025-11-24T10:30:00Z",
    "metric_score": 0.82,
    "version": "1.0.0"
  }
}
```

---

### 4. SignatureRegistry

The SignatureRegistry maps signature names to factory functions that create signature instances.
This allows modules to reference signatures by name in JSON while the actual types are compiled into the consumer's binary.

**Key Design:** ml-crate-dsrs provides the registry infrastructure; consumers register their own signatures.

```rust
use std::collections::HashMap;
use std::sync::Arc;
use dspy_rs::MetaSignature;

/// Factory function type for creating signature instances
type SignatureFactory = Box<dyn Fn() -> Box<dyn MetaSignature> + Send + Sync>;

/// Registry mapping signature names to factory functions
pub struct SignatureRegistry {
    factories: HashMap<String, SignatureFactory>,
}

impl SignatureRegistry {
    /// Create empty registry
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Register a signature type with a name
    ///
    /// # Example
    /// ```rust
    /// use dspy_rs::Signature;
    ///
    /// #[derive(Signature, Default)]
    /// struct NPCDialogue {
    ///     #[input] npc_personality: String,
    ///     #[input] player_message: String,
    ///     #[output] response: String,
    ///     #[output] emotion: String,
    /// }
    ///
    /// let mut registry = SignatureRegistry::new();
    /// registry.register::<NPCDialogue>("npc.dialogue");
    /// ```
    pub fn register<S>(&mut self, name: &str)
    where
        S: MetaSignature + Default + 'static,
    {
        self.factories.insert(
            name.to_string(),
            Box::new(|| Box::new(S::default()) as Box<dyn MetaSignature>),
        );
    }

    /// Create a signature instance by name
    pub fn create(&self, name: &str) -> Option<Box<dyn MetaSignature>> {
        self.factories.get(name).map(|factory| factory())
    }

    /// Check if a signature name is registered
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }

    /// List all registered signature names
    pub fn names(&self) -> Vec<&str> {
        self.factories.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for SignatureRegistry {
    fn default() -> Self {
        Self::new()
    }
}
```

**Consumer Usage Example:**

```rust
// In consumer's game project (not in ml-crate-dsrs)
use ml_crate_dsrs::{SignatureRegistry, DSPyEngine};
use dspy_rs::Signature;

// Define game-specific signatures
#[derive(Signature, Default)]
struct NPCDialogue {
    #[input] npc_personality: String,
    #[input] player_message: String,
    #[input] conversation_history: String,
    #[output] response: String,
    #[output] emotion: String,
}

#[derive(Signature, Default)]
struct MerchantHaggle {
    #[input] query: String,
    #[input] context: String,
    #[input] available_tools: String,
    #[output] response: String,
    #[output] tool_call: String,
}

#[derive(Signature, Default)]
struct QuestGenerate {
    #[input] player_level: String,
    #[input] location: String,
    #[input] recent_events: String,
    #[output] quest_title: String,
    #[output] quest_description: String,
    #[output] objectives: String,
}

fn main() {
    // Consumer creates and populates registry
    let mut registry = SignatureRegistry::new();
    registry.register::<NPCDialogue>("npc.dialogue");
    registry.register::<MerchantHaggle>("npc.merchant.haggle");
    registry.register::<QuestGenerate>("quest.generate");

    // Create engine with consumer's registry
    let engine = DSPyEngine::new(
        PathBuf::from("./modules"),
        adapter,
        Arc::new(registry),
    ).await?;

    // Now modules can reference signatures by name
    // e.g., module JSON has "signature_name": "npc.dialogue"
}
```

---

### 5. ModuleManifest

Central registry for fast module lookup.

```rust
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleManifest {
    pub version: String,
    pub modules: HashMap<String, ModuleEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleEntry {
    /// Relative path from modules/ directory
    pub path: String,

    /// File hash for hot reload detection
    pub hash: Option<String>,

    /// Tags for categorization
    pub tags: Vec<String>,
}
```

**Example manifest.json:**

```json
{
  "version": "1.0",
  "modules": {
    "npc.dialogue.casual": {
      "path": "npc/dialogue_casual.json",
      "hash": "sha256:abc123...",
      "tags": ["npc", "dialogue"]
    },
    "npc.merchant.haggle": {
      "path": "npc/merchant_haggle.json",
      "hash": "sha256:def456...",
      "tags": ["npc", "merchant", "tools"]
    },
    "quest.generate": {
      "path": "quest/generate.json",
      "hash": "sha256:ghi789...",
      "tags": ["quest", "generation"]
    }
  }
}
```

---

### 5. Tool System

#### Tool Trait

```rust
use async_trait::async_trait;
use serde_json::Value;

/// Abstracted tool trait - can wrap any function type
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name (must match what LLM outputs)
    fn name(&self) -> &str;

    /// Tool description (included in available_tools)
    fn description(&self) -> &str;

    /// JSON schema for arguments (for validation)
    fn args_schema(&self) -> Option<Value> {
        None
    }

    /// Execute the tool with given arguments
    async fn execute(&self, args: Value) -> Result<Value, ToolError>;
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("Tool not found: {0}")]
    NotFound(String),

    #[error("Invalid arguments: {0}")]
    InvalidArgs(String),

    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
}
```

#### ToolRegistry

```rust
use std::collections::HashMap;
use std::sync::Arc;

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: HashMap::new() }
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub async fn execute(&self, name: &str, args: Value) -> Result<Value, ToolError> {
        let tool = self.get(name).ok_or_else(|| ToolError::NotFound(name.to_string()))?;
        tool.execute(args).await
    }

    /// Generate JSON list for LLM context
    pub fn to_json(&self) -> Value {
        let tools: Vec<Value> = self.tools.values()
            .map(|t| serde_json::json!({
                "name": t.name(),
                "description": t.description(),
                "args_schema": t.args_schema()
            }))
            .collect();
        Value::Array(tools)
    }
}
```

#### RhaiTool Wrapper

Wraps a Rhai function as a Tool:

```rust
use rhai::{Engine, FnPtr, Dynamic};

pub struct RhaiTool {
    name: String,
    description: String,
    engine: Arc<Engine>,
    fn_ptr: FnPtr,
}

impl RhaiTool {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        engine: Arc<Engine>,
        fn_ptr: FnPtr,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            engine,
            fn_ptr,
        }
    }
}

#[async_trait]
impl Tool for RhaiTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    async fn execute(&self, args: Value) -> Result<Value, ToolError> {
        // Convert JSON args to Rhai Dynamic
        let rhai_args = json_to_dynamic(args)?;

        // Call Rhai function
        let result = self.fn_ptr.call::<Dynamic>(&self.engine, &[], rhai_args)
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        // Convert result back to JSON
        dynamic_to_json(result)
    }
}
```

---

### 6. ToolWrapper

Generic wrapper that adds tool-calling capability to any predictor.

```rust
use std::sync::Arc;
use tokio::sync::RwLock;

/// Tool call request parsed from LLM output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub args: Value,
}

/// Configuration for tool wrapper
pub struct ToolWrapperConfig {
    pub max_iterations: usize,
    pub tool_result_key: String,  // Key to inject tool results
}

impl Default for ToolWrapperConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            tool_result_key: "context".to_string(),
        }
    }
}

/// Generic tool wrapper for any predictor type
pub struct ToolWrapper {
    tools: Arc<ToolRegistry>,
    config: ToolWrapperConfig,
}

impl ToolWrapper {
    pub fn new(tools: Arc<ToolRegistry>, config: ToolWrapperConfig) -> Self {
        Self { tools, config }
    }

    /// Invoke module with tool support
    pub async fn invoke(
        &self,
        engine: &DSPyEngine,
        module_id: &str,
        mut input: Value,
    ) -> Result<Value, DSPyEngineError> {
        // Inject available tools into input
        input["available_tools"] = self.tools.to_json();

        for iteration in 0..self.config.max_iterations {
            // Call the underlying predictor
            let output = engine.invoke_raw(module_id, input.clone()).await?;

            // Check for tool call in output
            match output.get("tool_call") {
                Some(tool_call) if !tool_call.is_null() => {
                    let call: ToolCall = serde_json::from_value(tool_call.clone())
                        .map_err(|e| DSPyEngineError::ParseError(e.to_string()))?;

                    // Execute the tool
                    let result = self.tools.execute(&call.name, call.args).await
                        .map_err(|e| DSPyEngineError::ToolError(e))?;

                    // Inject result into context for next iteration
                    let context_update = format!(
                        "Tool '{}' returned: {}",
                        call.name,
                        serde_json::to_string(&result).unwrap_or_default()
                    );

                    // Append to existing context or create new
                    let existing = input.get(&self.config.tool_result_key)
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    input[&self.config.tool_result_key] = Value::String(
                        if existing.is_empty() {
                            context_update
                        } else {
                            format!("{}\n{}", existing, context_update)
                        }
                    );
                }
                _ => {
                    // No tool call - return the response
                    return Ok(output);
                }
            }
        }

        Err(DSPyEngineError::MaxIterationsReached(self.config.max_iterations))
    }
}
```

---

### 7. DSPyEngine

Main orchestrator.

```rust
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use notify::{Watcher, RecursiveMode};

pub struct DSPyEngine {
    /// Loaded modules
    modules: Arc<RwLock<HashMap<String, OptimizedModule>>>,

    /// Module manifest
    manifest: Arc<RwLock<ModuleManifest>>,

    /// Modules directory path
    modules_dir: PathBuf,

    /// CandleAdapter for inference
    adapter: Arc<CandleAdapter>,

    /// Signature registry (provided by consumer)
    signature_registry: Arc<SignatureRegistry>,

    /// Tool registry (shared with ToolWrapper)
    tools: Arc<ToolRegistry>,

    /// Tool wrapper for tool-enabled invocations
    tool_wrapper: ToolWrapper,

    /// Tokio runtime for async operations
    runtime: Arc<tokio::runtime::Runtime>,
}

impl DSPyEngine {
    /// Create new engine and load all modules
    ///
    /// # Arguments
    /// * `modules_dir` - Path to directory containing module JSON files and manifest.json
    /// * `adapter` - CandleAdapter instance for local GPU inference
    /// * `signature_registry` - Consumer-provided registry mapping signature names to types
    ///
    /// # Example
    /// ```rust
    /// // Consumer creates and populates the signature registry
    /// let mut registry = SignatureRegistry::new();
    /// registry.register::<NPCDialogue>("npc.dialogue");
    /// registry.register::<MerchantHaggle>("npc.merchant.haggle");
    ///
    /// // Pass registry to engine
    /// let engine = DSPyEngine::new(
    ///     PathBuf::from("./modules"),
    ///     adapter,
    ///     Arc::new(registry),
    /// ).await?;
    /// ```
    pub async fn new(
        modules_dir: PathBuf,
        adapter: Arc<CandleAdapter>,
        signature_registry: Arc<SignatureRegistry>,
    ) -> Result<Self, DSPyEngineError> {
        // REQUIRED: Initialize dspy-rs global state before any predictor calls
        // configure() takes BOTH lm AND adapter as separate arguments
        let lm = dspy_rs::LM::builder()
            .model("local".to_string())  // Placeholder - adapter handles real inference
            .build()
            .await
            .map_err(|e| DSPyEngineError::RuntimeError(format!("Failed to create LM: {}", e)))?;

        // Configure global settings with LM and our CandleAdapter
        dspy_rs::configure(lm, adapter.as_ref().clone());

        let tools = Arc::new(ToolRegistry::new());
        let tool_wrapper = ToolWrapper::new(tools.clone(), ToolWrapperConfig::default());

        let engine = Self {
            modules: Arc::new(RwLock::new(HashMap::new())),
            manifest: Arc::new(RwLock::new(ModuleManifest::default())),
            modules_dir,
            adapter,
            signature_registry,
            tools,
            tool_wrapper,
            runtime: Arc::new(tokio::runtime::Runtime::new()?),
        };

        // Load manifest and all modules
        engine.reload_all().await?;

        Ok(engine)
    }

    /// Load/reload all modules from disk
    pub async fn reload_all(&self) -> Result<(), DSPyEngineError> {
        let manifest_path = self.modules_dir.join("manifest.json");
        let manifest: ModuleManifest = load_json(&manifest_path)?;

        let mut modules = self.modules.write().await;
        modules.clear();

        for (id, entry) in &manifest.modules {
            let module_path = self.modules_dir.join(&entry.path);
            let module: OptimizedModule = load_json(&module_path)?;
            modules.insert(id.clone(), module);
        }

        *self.manifest.write().await = manifest;

        Ok(())
    }

    /// Reload a single module by ID
    pub async fn reload_module(&self, module_id: &str) -> Result<(), DSPyEngineError> {
        let manifest = self.manifest.read().await;
        let entry = manifest.modules.get(module_id)
            .ok_or_else(|| DSPyEngineError::ModuleNotFound(module_id.to_string()))?;

        let module_path = self.modules_dir.join(&entry.path);
        let module: OptimizedModule = load_json(&module_path)?;

        self.modules.write().await.insert(module_id.to_string(), module);

        Ok(())
    }

    /// Enable hot reload via file watcher
    pub fn enable_hot_reload(&self) -> Result<(), DSPyEngineError> {
        let modules = self.modules.clone();
        let manifest = self.manifest.clone();
        let modules_dir = self.modules_dir.clone();

        let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            if let Ok(event) = res {
                if event.paths.iter().any(|p| p.extension().map(|e| e == "json").unwrap_or(false)) {
                    // Trigger reload (simplified - production would be more sophisticated)
                    tracing::info!("Module file changed, reloading...");
                    // Note: Would need to spawn async task here
                }
            }
        })?;

        watcher.watch(&self.modules_dir, RecursiveMode::Recursive)?;

        // Keep watcher alive (in production, store in self)
        std::mem::forget(watcher);

        Ok(())
    }

    /// Register a tool
    pub fn register_tool(&self, tool: Arc<dyn Tool>) {
        // Note: Would need interior mutability for ToolRegistry
        // This is simplified - production would use RwLock
    }

    /// Get module by ID
    pub async fn get_module(&self, module_id: &str) -> Option<OptimizedModule> {
        self.modules.read().await.get(module_id).cloned()
    }

    /// Invoke module without tool support (async)
    pub async fn invoke(
        &self,
        module_id: &str,
        input: Value,
    ) -> Result<Value, DSPyEngineError> {
        self.invoke_raw(module_id, input).await
    }

    /// Invoke module with tool support (async)
    pub async fn invoke_with_tools(
        &self,
        module_id: &str,
        input: Value,
    ) -> Result<Value, DSPyEngineError> {
        let module = self.get_module(module_id).await
            .ok_or_else(|| DSPyEngineError::ModuleNotFound(module_id.to_string()))?;

        if !module.tool_enabled {
            return Err(DSPyEngineError::ToolsNotEnabled(module_id.to_string()));
        }

        self.tool_wrapper.invoke(self, module_id, input).await
    }

    /// Sync wrapper for Rhai (blocks on async)
    pub fn invoke_sync(
        &self,
        module_id: &str,
        input: Value,
    ) -> Result<Value, DSPyEngineError> {
        self.runtime.block_on(self.invoke(module_id, input))
    }

    /// Sync wrapper with tools for Rhai
    pub fn invoke_with_tools_sync(
        &self,
        module_id: &str,
        input: Value,
    ) -> Result<Value, DSPyEngineError> {
        self.runtime.block_on(self.invoke_with_tools(module_id, input))
    }

    /// Internal: Raw invocation (reconstructs predictor and calls it)
    async fn invoke_raw(
        &self,
        module_id: &str,
        input: Value,
    ) -> Result<Value, DSPyEngineError> {
        let module = self.get_module(module_id).await
            .ok_or_else(|| DSPyEngineError::ModuleNotFound(module_id.to_string()))?;

        // Reconstruct predictor from optimized module
        let output = match module.predictor_type {
            PredictorType::Predict => {
                self.execute_predict(&module, input).await?
            }
            PredictorType::ChainOfThought => {
                self.execute_chain_of_thought(&module, input).await?
            }
        };

        Ok(output)
    }

    /// Execute with Predict predictor
    async fn execute_predict(
        &self,
        module: &OptimizedModule,
        input: Value,
    ) -> Result<Value, DSPyEngineError> {
        // Build prompt from module instruction, demos, and input
        let prompt = self.build_prompt(module, &input)?;

        // Call CandleAdapter
        // Note: This is simplified - production would use dspy-rs Predict properly
        let response = self.adapter.generate(&prompt).await
            .map_err(|e| DSPyEngineError::InferenceError(e.to_string()))?;

        // Parse response into output fields
        self.parse_output(module, &response)
    }

    /// Execute with ChainOfThought predictor
    async fn execute_chain_of_thought(
        &self,
        module: &OptimizedModule,
        input: Value,
    ) -> Result<Value, DSPyEngineError> {
        // Similar to Predict but includes reasoning field
        // Note: Simplified - production would use dspy-rs ChainOfThought
        let prompt = self.build_cot_prompt(module, &input)?;

        let response = self.adapter.generate(&prompt).await
            .map_err(|e| DSPyEngineError::InferenceError(e.to_string()))?;

        self.parse_cot_output(module, &response)
    }

    // Helper methods for prompt building and output parsing
    fn build_prompt(&self, module: &OptimizedModule, input: &Value) -> Result<String, DSPyEngineError> {
        // TODO: Implement prompt building from module + input
        // Should use module.instruction, module.demos, and input fields
        todo!()
    }

    fn build_cot_prompt(&self, module: &OptimizedModule, input: &Value) -> Result<String, DSPyEngineError> {
        // TODO: Implement CoT prompt building (includes reasoning step)
        todo!()
    }

    fn parse_output(&self, module: &OptimizedModule, response: &str) -> Result<Value, DSPyEngineError> {
        // TODO: Parse LLM response into output fields defined by signature
        todo!()
    }

    fn parse_cot_output(&self, module: &OptimizedModule, response: &str) -> Result<Value, DSPyEngineError> {
        // TODO: Parse CoT response (includes reasoning + output fields)
        todo!()
    }
}
```

---

### 8. Error Types

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DSPyEngineError {
    #[error("Module not found: {0}")]
    ModuleNotFound(String),

    #[error("Tools not enabled for module: {0}")]
    ToolsNotEnabled(String),

    #[error("Max iterations reached: {0}")]
    MaxIterationsReached(usize),

    #[error("Tool error: {0}")]
    ToolError(#[from] ToolError),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("File watcher error: {0}")]
    WatcherError(#[from] notify::Error),

    #[error("Runtime error: {0}")]
    RuntimeError(String),
}
```

---

## Rhai Integration

### Registering DSPyEngine in Rhai

```rust
use rhai::{Engine, Dynamic, Map};

pub fn register_dspy_engine(
    rhai_engine: &mut Engine,
    dspy_engine: Arc<DSPyEngine>,
) {
    let engine_clone = dspy_engine.clone();
    rhai_engine.register_fn("invoke", move |module_id: &str, input: Map| -> Dynamic {
        let input_json = map_to_json(input);
        match engine_clone.invoke_sync(module_id, input_json) {
            Ok(result) => json_to_dynamic(result),
            Err(e) => Dynamic::from(format!("Error: {}", e)),
        }
    });

    let engine_clone = dspy_engine.clone();
    rhai_engine.register_fn("invoke_with_tools", move |module_id: &str, input: Map| -> Dynamic {
        let input_json = map_to_json(input);
        match engine_clone.invoke_with_tools_sync(module_id, input_json) {
            Ok(result) => json_to_dynamic(result),
            Err(e) => Dynamic::from(format!("Error: {}", e)),
        }
    });

    // Register tool registration function
    let engine_clone = dspy_engine.clone();
    rhai_engine.register_fn("register_tool", move |name: &str, desc: &str, fn_ptr: FnPtr| {
        let tool = RhaiTool::new(name, desc, /* engine */, fn_ptr);
        engine_clone.register_tool(Arc::new(tool));
    });
}
```

### Example Rhai Script

```rhai
// Register game tools
register_tool("get_player_gold", "Get player's current gold amount", || {
    // This function has access to game state
    player.gold
});

register_tool("get_inventory", "Get player's inventory", || {
    player.inventory.to_json()
});

register_tool("get_npc_mood", "Get NPC's current mood", |npc_id| {
    game.npcs[npc_id].mood
});

// Simple invocation (no tools)
fn handle_npc_greeting(npc_id, player_message) {
    let npc = game.npcs[npc_id];

    let result = llm_manager.invoke("npc.dialogue.casual", #{
        npc_personality: npc.personality,
        player_message: player_message,
        conversation_history: npc.conversation_history
    });

    return result.response;
}

// Tool-enabled invocation
fn handle_merchant_interaction(merchant_id, player_message) {
    let merchant = game.npcs[merchant_id];

    let result = llm_manager.invoke_with_tools("npc.merchant.haggle", #{
        query: player_message,
        context: ""
    });

    return result.response;
}
```

---

## File Structure

```
modules/
├── manifest.json
├── npc/
│   ├── dialogue_casual.json
│   ├── dialogue_formal.json
│   └── merchant_haggle.json
├── quest/
│   ├── generate.json
│   └── hints.json
└── combat/
    └── narration.json

src/
├── inference/
│   ├── mod.rs              # Module exports
│   ├── engine.rs           # DSPyEngine implementation
│   ├── module.rs           # OptimizedModule, SignatureDefinition
│   ├── manifest.rs         # ModuleManifest
│   ├── registry.rs         # SignatureRegistry (consumer registers signatures here)
│   └── error.rs            # DSPyEngineError
├── tools/
│   ├── mod.rs              # Module exports
│   ├── traits.rs           # Tool trait
│   ├── registry.rs         # ToolRegistry
│   ├── wrapper.rs          # ToolWrapper
│   └── rhai_tool.rs        # RhaiTool implementation
└── rhai/
    ├── mod.rs              # Rhai integration
    └── registration.rs     # Engine registration helpers
```

---

## Dependencies

```toml
[dependencies]
# Existing
dspy-rs = "0.7.3"
candle-core = { version = "0.9", features = ["cuda"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"
async-trait = "0.1"
tracing = "0.1"

# New for DSPy Engine
rhai = "1.0"                    # Rhai scripting engine
notify = "6.0"                  # File watching for hot reload
```

---

## Testing Strategy

### Unit Tests (Rust)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimized_module_deserialization() {
        let json = r#"{
            "module_id": "test.module",
            "predictor_type": "predict",
            ...
        }"#;
        let module: OptimizedModule = serde_json::from_str(json).unwrap();
        assert_eq!(module.module_id, "test.module");
    }

    #[test]
    fn test_tool_registry() {
        let mut registry = ToolRegistry::new();
        // Register mock tool
        // Assert execution works
    }

    #[tokio::test]
    async fn test_tool_wrapper_no_tool_call() {
        // Test that wrapper returns immediately when no tool_call in output
    }

    #[tokio::test]
    async fn test_tool_wrapper_with_tool_call() {
        // Test that wrapper executes tool and re-invokes
    }

    #[tokio::test]
    async fn test_tool_wrapper_max_iterations() {
        // Test that wrapper stops at max iterations
    }
}
```

### Integration Tests (Rhai Scripts)

```rhai
// test_scripts/test_simple_invoke.rhai
let result = llm_manager.invoke("npc.dialogue.casual", #{
    npc_personality: "friendly",
    player_message: "Hello!",
    conversation_history: ""
});

assert(result.response != "", "Should return non-empty response");
assert(result.emotion != "", "Should return emotion");
```

```rhai
// test_scripts/test_tool_invoke.rhai
register_tool("mock_tool", "Returns fixed value", || {
    "mock_result"
});

let result = llm_manager.invoke_with_tools("test.tool_module", #{
    query: "Use the tool"
});

assert(result.response.contains("mock_result"), "Should include tool result");
```

---

## Success Criteria

### Phase 3A: Core Engine ✅ COMPLETE
- [x] OptimizedModule deserializes from JSON (with signature_name field)
- [x] ModuleManifest loads and indexes modules
- [x] SignatureRegistry implemented (consumer registration pattern)
- [x] dspy_rs::configure(lm, adapter) called on engine initialization
- [x] Value ↔ Example conversion helpers implemented
- [x] DSPyEngine.invoke() works with Predict modules
- [x] DSPyEngine.invoke() works with ChainOfThought modules
- [x] Unit tests pass (63 passing + 10 ignored)
- [x] Integration tests pass (10 passing + 6 model tests verified)

### Phase 3A-Optional: Hot Reload
- [ ] File watcher detects JSON changes
- [ ] Module reloads without restart
- [ ] Hot reload tests pass

### Phase 3B: Tool System
- [ ] Tool trait implemented
- [ ] ToolRegistry stores and retrieves tools
- [ ] ToolRegistry.execute() calls tool with args
- [ ] ToolWrapper parses tool_call from LLM output
- [ ] ToolWrapper executes tool and feeds result back
- [ ] ToolWrapper re-invokes predictor with tool result
- [ ] Max iterations limit prevents infinite loops
- [ ] Tool unit tests pass

### Phase 3C: Rhai Integration
- [ ] DSPyEngine registered as Rhai type
- [ ] invoke() callable from Rhai scripts
- [ ] invoke_with_tools() callable from Rhai scripts
- [ ] RhaiTool wraps Rhai FnPtr as Tool trait
- [ ] register_tool() callable from Rhai scripts
- [ ] JSON ↔ Dynamic conversion helpers work
- [ ] End-to-end Rhai integration tests pass

---

## Future Enhancements (Phase 4+)

- [ ] ReAct predictor (when dspy-rs supports it)
- [ ] Module versioning and rollback
- [ ] Metrics and telemetry
- [ ] Module warmup/preloading
- [ ] Tool response caching
- [ ] Parallel tool execution
- [ ] Streaming responses

---

## References

### Verified Against
- dspy-rs v0.7.3 source: `.claude/knowledge/dspy/source/`
- CandleAdapter spec: `specs/01-candle-adapter.md`
- Architecture: `ARCH.md`

### Related Documentation
- [ARCH.md](../ARCH.md) - Overall system architecture
- [01-candle-adapter.md](./01-candle-adapter.md) - CandleAdapter specification
