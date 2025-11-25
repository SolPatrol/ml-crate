//! Module types for pre-optimized DSPy modules
//!
//! Defines the data structures for loading optimized DSPy modules from JSON files.
//! These modules contain pre-optimized instructions and demonstrations that were
//! created by DSPy optimizers like MIPROv2 or COPRO.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Predictor execution strategy
///
/// Determines how the module processes inputs and generates outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictorType {
    /// Simple predict - direct input → output mapping
    #[default]
    Predict,
    /// Chain of thought - includes reasoning step before output
    ChainOfThought,
    // ReAct - TODO: Phase 4 (requires upstream dspy-rs support)
}

impl std::fmt::Display for PredictorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Predict => write!(f, "predict"),
            Self::ChainOfThought => write!(f, "chain_of_thought"),
        }
    }
}

/// Few-shot demonstration example
///
/// Demonstrations are input→output pairs that show the model how to respond.
/// They are typically created by DSPy optimizers like BootstrapFewShot or MIPROv2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Demo {
    /// Input field values for this demonstration
    pub inputs: HashMap<String, Value>,
    /// Expected output field values for this demonstration
    pub outputs: HashMap<String, Value>,
}

impl Demo {
    /// Create a new demonstration
    pub fn new(inputs: HashMap<String, Value>, outputs: HashMap<String, Value>) -> Self {
        Self { inputs, outputs }
    }

    /// Create an empty demonstration
    pub fn empty() -> Self {
        Self {
            inputs: HashMap::new(),
            outputs: HashMap::new(),
        }
    }

    /// Get an input value by key
    pub fn get_input(&self, key: &str) -> Option<&Value> {
        self.inputs.get(key)
    }

    /// Get an output value by key
    pub fn get_output(&self, key: &str) -> Option<&Value> {
        self.outputs.get(key)
    }
}

/// Field definition for signature metadata
///
/// Describes a single input or output field in a signature.
/// This is documentation/metadata only - the actual signature type
/// comes from the SignatureRegistry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    /// Field name (e.g., "question", "answer")
    pub name: String,
    /// Human-readable description of the field
    #[serde(default)]
    pub description: Option<String>,
    /// Field type hint (e.g., "string", "json")
    #[serde(default = "default_field_type")]
    pub field_type: String,
}

fn default_field_type() -> String {
    "string".to_string()
}

impl FieldDefinition {
    /// Create a new field definition
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            field_type: "string".to_string(),
        }
    }

    /// Create a new field definition with description
    pub fn with_description(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: Some(description.into()),
            field_type: "string".to_string(),
        }
    }

    /// Set the field type
    pub fn with_type(mut self, field_type: impl Into<String>) -> Self {
        self.field_type = field_type.into();
        self
    }
}

/// Signature definition for documentation/validation
///
/// This is metadata about the signature - the actual signature type
/// comes from the SignatureRegistry. This allows tools and documentation
/// to understand the module's interface without needing the compiled type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SignatureDefinition {
    /// Input field definitions
    #[serde(default)]
    pub inputs: Vec<FieldDefinition>,
    /// Output field definitions
    #[serde(default)]
    pub outputs: Vec<FieldDefinition>,
}

impl SignatureDefinition {
    /// Create a new empty signature definition
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a signature definition with inputs and outputs
    pub fn with_fields(inputs: Vec<FieldDefinition>, outputs: Vec<FieldDefinition>) -> Self {
        Self { inputs, outputs }
    }

    /// Add an input field
    pub fn add_input(&mut self, field: FieldDefinition) {
        self.inputs.push(field);
    }

    /// Add an output field
    pub fn add_output(&mut self, field: FieldDefinition) {
        self.outputs.push(field);
    }

    /// Get input field names
    pub fn input_names(&self) -> Vec<&str> {
        self.inputs.iter().map(|f| f.name.as_str()).collect()
    }

    /// Get output field names
    pub fn output_names(&self) -> Vec<&str> {
        self.outputs.iter().map(|f| f.name.as_str()).collect()
    }
}

/// Module metadata for debugging and versioning
///
/// Contains information about how and when the module was optimized.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleMetadata {
    /// Optimizer that created this module (e.g., "MIPROv2", "COPRO", "manual")
    pub optimizer: String,
    /// ISO 8601 timestamp of when the module was optimized
    #[serde(default)]
    pub optimized_at: Option<String>,
    /// Optimization metric score (if available)
    #[serde(default)]
    pub metric_score: Option<f32>,
    /// Module version string
    pub version: String,
}

impl Default for ModuleMetadata {
    fn default() -> Self {
        Self {
            optimizer: "manual".to_string(),
            optimized_at: None,
            metric_score: None,
            version: "1.0.0".to_string(),
        }
    }
}

impl ModuleMetadata {
    /// Create metadata for a manually created module
    pub fn manual(version: impl Into<String>) -> Self {
        Self {
            optimizer: "manual".to_string(),
            optimized_at: None,
            metric_score: None,
            version: version.into(),
        }
    }

    /// Create metadata for an optimized module
    pub fn optimized(
        optimizer: impl Into<String>,
        version: impl Into<String>,
        score: Option<f32>,
    ) -> Self {
        Self {
            optimizer: optimizer.into(),
            optimized_at: Some(chrono_timestamp()),
            metric_score: score,
            version: version.into(),
        }
    }
}

/// Get current timestamp in ISO 8601 format
fn chrono_timestamp() -> String {
    // Simple timestamp without chrono dependency
    // In production, use chrono crate for proper formatting
    "".to_string()
}

/// Pre-optimized DSPy module loaded from JSON
///
/// Represents a module that has been optimized by a DSPy optimizer
/// (MIPROv2, COPRO, etc.) and saved to disk for runtime loading.
///
/// # Example JSON Format
///
/// ```json
/// {
///   "module_id": "npc.dialogue.casual",
///   "predictor_type": "predict",
///   "signature_name": "npc.dialogue",
///   "instruction": "You are roleplaying as an NPC...",
///   "demos": [
///     {
///       "inputs": { "npc_personality": "gruff", "player_message": "Hello" },
///       "outputs": { "response": "*grunts*", "emotion": "annoyed" }
///     }
///   ],
///   "tool_enabled": false,
///   "metadata": { "optimizer": "MIPROv2", "version": "1.0.0" }
/// }
/// ```
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
    #[serde(default)]
    pub demos: Vec<Demo>,

    /// Whether this module can request tool calls
    #[serde(default)]
    pub tool_enabled: bool,

    /// Metadata for debugging and versioning
    #[serde(default)]
    pub metadata: ModuleMetadata,
}

impl OptimizedModule {
    /// Create a new optimized module with minimal configuration
    pub fn new(
        module_id: impl Into<String>,
        signature_name: impl Into<String>,
        instruction: impl Into<String>,
    ) -> Self {
        Self {
            module_id: module_id.into(),
            predictor_type: PredictorType::default(),
            signature_name: signature_name.into(),
            signature: None,
            instruction: instruction.into(),
            demos: Vec::new(),
            tool_enabled: false,
            metadata: ModuleMetadata::default(),
        }
    }

    /// Set the predictor type
    pub fn with_predictor_type(mut self, predictor_type: PredictorType) -> Self {
        self.predictor_type = predictor_type;
        self
    }

    /// Add demonstrations
    pub fn with_demos(mut self, demos: Vec<Demo>) -> Self {
        self.demos = demos;
        self
    }

    /// Enable tool support
    pub fn with_tools_enabled(mut self) -> Self {
        self.tool_enabled = true;
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: ModuleMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set signature definition
    pub fn with_signature(mut self, signature: SignatureDefinition) -> Self {
        self.signature = Some(signature);
        self
    }

    /// Check if this module uses chain of thought
    pub fn is_chain_of_thought(&self) -> bool {
        matches!(self.predictor_type, PredictorType::ChainOfThought)
    }

    /// Get the number of demonstrations
    pub fn demo_count(&self) -> usize {
        self.demos.len()
    }

    /// Check if signature definition is available
    pub fn has_signature_definition(&self) -> bool {
        self.signature.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_predictor_type_serialization() {
        let predict = PredictorType::Predict;
        assert_eq!(
            serde_json::to_string(&predict).unwrap(),
            "\"predict\""
        );

        let cot = PredictorType::ChainOfThought;
        assert_eq!(
            serde_json::to_string(&cot).unwrap(),
            "\"chain_of_thought\""
        );
    }

    #[test]
    fn test_predictor_type_deserialization() {
        let predict: PredictorType = serde_json::from_str("\"predict\"").unwrap();
        assert_eq!(predict, PredictorType::Predict);

        let cot: PredictorType = serde_json::from_str("\"chain_of_thought\"").unwrap();
        assert_eq!(cot, PredictorType::ChainOfThought);
    }

    #[test]
    fn test_demo_creation() {
        let mut inputs = HashMap::new();
        inputs.insert("question".to_string(), json!("What is 2+2?"));

        let mut outputs = HashMap::new();
        outputs.insert("answer".to_string(), json!("4"));

        let demo = Demo::new(inputs, outputs);

        assert_eq!(
            demo.get_input("question").unwrap().as_str().unwrap(),
            "What is 2+2?"
        );
        assert_eq!(demo.get_output("answer").unwrap().as_str().unwrap(), "4");
    }

    #[test]
    fn test_field_definition() {
        let field = FieldDefinition::new("question")
            .with_type("string");

        assert_eq!(field.name, "question");
        assert_eq!(field.field_type, "string");
        assert!(field.description.is_none());

        let field_with_desc = FieldDefinition::with_description("answer", "The computed answer");
        assert_eq!(field_with_desc.name, "answer");
        assert_eq!(field_with_desc.description.unwrap(), "The computed answer");
    }

    #[test]
    fn test_signature_definition() {
        let mut sig = SignatureDefinition::new();
        sig.add_input(FieldDefinition::new("question"));
        sig.add_output(FieldDefinition::new("answer"));

        assert_eq!(sig.input_names(), vec!["question"]);
        assert_eq!(sig.output_names(), vec!["answer"]);
    }

    #[test]
    fn test_module_metadata_default() {
        let meta = ModuleMetadata::default();
        assert_eq!(meta.optimizer, "manual");
        assert_eq!(meta.version, "1.0.0");
        assert!(meta.optimized_at.is_none());
        assert!(meta.metric_score.is_none());
    }

    #[test]
    fn test_optimized_module_creation() {
        let module = OptimizedModule::new(
            "test.module",
            "test.signature",
            "Answer the question.",
        )
        .with_predictor_type(PredictorType::ChainOfThought)
        .with_tools_enabled();

        assert_eq!(module.module_id, "test.module");
        assert_eq!(module.signature_name, "test.signature");
        assert!(module.is_chain_of_thought());
        assert!(module.tool_enabled);
    }

    #[test]
    fn test_optimized_module_deserialization() {
        let json = r#"{
            "module_id": "npc.dialogue.casual",
            "predictor_type": "predict",
            "signature_name": "npc.dialogue",
            "instruction": "You are an NPC.",
            "demos": [
                {
                    "inputs": { "player_message": "Hello" },
                    "outputs": { "response": "Hi there!" }
                }
            ],
            "tool_enabled": false,
            "metadata": {
                "optimizer": "MIPROv2",
                "optimized_at": "2025-11-24T10:30:00Z",
                "metric_score": 0.87,
                "version": "1.0.0"
            }
        }"#;

        let module: OptimizedModule = serde_json::from_str(json).unwrap();

        assert_eq!(module.module_id, "npc.dialogue.casual");
        assert_eq!(module.predictor_type, PredictorType::Predict);
        assert_eq!(module.signature_name, "npc.dialogue");
        assert_eq!(module.instruction, "You are an NPC.");
        assert_eq!(module.demos.len(), 1);
        assert!(!module.tool_enabled);
        assert_eq!(module.metadata.optimizer, "MIPROv2");
        assert_eq!(module.metadata.metric_score, Some(0.87));
    }

    #[test]
    fn test_optimized_module_serialization_roundtrip() {
        let original = OptimizedModule::new(
            "test.module",
            "test.sig",
            "Test instruction",
        )
        .with_predictor_type(PredictorType::ChainOfThought)
        .with_demos(vec![Demo::new(
            [("input".to_string(), json!("value"))].into_iter().collect(),
            [("output".to_string(), json!("result"))].into_iter().collect(),
        )]);

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: OptimizedModule = serde_json::from_str(&json).unwrap();

        assert_eq!(original.module_id, deserialized.module_id);
        assert_eq!(original.predictor_type, deserialized.predictor_type);
        assert_eq!(original.demos.len(), deserialized.demos.len());
    }

    #[test]
    fn test_module_with_signature_definition() {
        let json = r#"{
            "module_id": "qa.simple",
            "predictor_type": "predict",
            "signature_name": "qa",
            "signature": {
                "inputs": [
                    { "name": "question", "description": "The question to answer", "field_type": "string" }
                ],
                "outputs": [
                    { "name": "answer", "description": "The answer", "field_type": "string" }
                ]
            },
            "instruction": "Answer concisely.",
            "metadata": { "optimizer": "manual", "version": "1.0.0" }
        }"#;

        let module: OptimizedModule = serde_json::from_str(json).unwrap();

        assert!(module.has_signature_definition());
        let sig = module.signature.unwrap();
        assert_eq!(sig.inputs.len(), 1);
        assert_eq!(sig.outputs.len(), 1);
        assert_eq!(sig.inputs[0].name, "question");
    }

    #[test]
    fn test_module_without_optional_fields() {
        // Test that modules can be loaded without optional fields
        let json = r#"{
            "module_id": "minimal",
            "predictor_type": "predict",
            "signature_name": "minimal",
            "instruction": "Minimal module."
        }"#;

        let module: OptimizedModule = serde_json::from_str(json).unwrap();

        assert_eq!(module.module_id, "minimal");
        assert!(module.demos.is_empty());
        assert!(!module.tool_enabled);
        assert_eq!(module.metadata.optimizer, "manual"); // default
    }
}
