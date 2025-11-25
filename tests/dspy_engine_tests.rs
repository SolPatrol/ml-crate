//! Integration tests for DSPyEngine
//!
//! These tests verify the DSPyEngine can:
//! - Load modules from disk
//! - Execute Predict modules
//! - Execute ChainOfThought modules
//! - Handle errors correctly
//!
//! Note: Tests marked with #[ignore] require a real model loaded via ModelPool.
//! Run with `cargo test -- --ignored` to include these tests.

use std::path::PathBuf;
use std::sync::Arc;

use dspy_rs::{Example, MetaSignature};
use serde_json::{json, Value};

use ml_crate_dsrs::inference::{
    conversion::demos_to_examples,
    manifest::{load_json, ModuleEntry, ModuleManifest},
    module::{Demo, OptimizedModule, PredictorType},
    registry::SignatureRegistry,
    DSPyEngine, DSPyEngineError,
};
use ml_crate_dsrs::{CandleAdapter, CandleConfig, ModelPool};

/// Test signature for integration tests
struct TestQASignature {
    instruction: String,
    demos: Vec<Example>,
}

impl Default for TestQASignature {
    fn default() -> Self {
        Self {
            instruction: "Answer the question.".to_string(),
            demos: Vec::new(),
        }
    }
}

impl MetaSignature for TestQASignature {
    fn demos(&self) -> Vec<Example> {
        self.demos.clone()
    }

    fn set_demos(&mut self, demos: Vec<Example>) -> anyhow::Result<()> {
        self.demos = demos;
        Ok(())
    }

    fn instruction(&self) -> String {
        self.instruction.clone()
    }

    fn input_fields(&self) -> Value {
        json!(["question"])
    }

    fn output_fields(&self) -> Value {
        json!(["answer"])
    }

    fn update_instruction(&mut self, instruction: String) -> anyhow::Result<()> {
        self.instruction = instruction;
        Ok(())
    }

    fn append(&mut self, _name: &str, _value: Value) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Get path to test fixtures directory
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("modules")
}

#[test]
fn test_load_manifest() {
    let manifest_path = fixtures_dir().join("manifest.json");
    let manifest: ModuleManifest = load_json(&manifest_path).unwrap();

    assert_eq!(manifest.version, "1.0");
    assert!(manifest.contains("test.predict"));
    assert!(manifest.contains("test.cot"));
    assert!(manifest.contains("qa.simple"));
}

#[test]
fn test_load_predict_module() {
    let module_path = fixtures_dir().join("test_predict.json");
    let module: OptimizedModule = load_json(&module_path).unwrap();

    assert_eq!(module.module_id, "test.predict");
    assert_eq!(module.predictor_type, PredictorType::Predict);
    assert_eq!(module.signature_name, "test.qa");
    assert!(!module.instruction.is_empty());
    assert_eq!(module.demos.len(), 2);
    assert!(!module.tool_enabled);
}

#[test]
fn test_load_cot_module() {
    let module_path = fixtures_dir().join("test_cot.json");
    let module: OptimizedModule = load_json(&module_path).unwrap();

    assert_eq!(module.module_id, "test.cot");
    assert_eq!(module.predictor_type, PredictorType::ChainOfThought);
    assert!(module.is_chain_of_thought());
    assert_eq!(module.demos.len(), 1);
}

#[test]
fn test_demo_to_example_conversion() {
    let demo = Demo::new(
        [("question".to_string(), json!("What is 2+2?"))]
            .into_iter()
            .collect(),
        [("answer".to_string(), json!("4"))]
            .into_iter()
            .collect(),
    );

    let examples = demos_to_examples(&[demo]);
    assert_eq!(examples.len(), 1);

    let example = &examples[0];
    assert!(example.input_keys.contains(&"question".to_string()));
    assert!(example.output_keys.contains(&"answer".to_string()));
}

#[test]
fn test_signature_registry() {
    let mut registry = SignatureRegistry::new();
    registry.register::<TestQASignature>("test.qa");

    assert!(registry.contains("test.qa"));
    assert!(!registry.contains("nonexistent"));

    let signature = registry.create("test.qa");
    assert!(signature.is_some());

    let sig = signature.unwrap();
    assert_eq!(sig.input_fields(), json!(["question"]));
    assert_eq!(sig.output_fields(), json!(["answer"]));
}

#[test]
fn test_module_metadata() {
    let module_path = fixtures_dir().join("qa_simple.json");
    let module: OptimizedModule = load_json(&module_path).unwrap();

    assert_eq!(module.metadata.optimizer, "MIPROv2");
    assert!(module.metadata.optimized_at.is_some());
    assert_eq!(module.metadata.metric_score, Some(0.85));
    assert_eq!(module.metadata.version, "1.0.0");
}

#[test]
fn test_module_entry_tags() {
    let manifest_path = fixtures_dir().join("manifest.json");
    let manifest: ModuleManifest = load_json(&manifest_path).unwrap();

    let predict_entry = manifest.get("test.predict").unwrap();
    assert!(predict_entry.has_tag("test"));
    assert!(predict_entry.has_tag("predict"));
    assert!(!predict_entry.has_tag("nonexistent"));

    let test_modules = manifest.modules_with_tag("test");
    assert!(test_modules.contains(&"test.predict"));
    assert!(test_modules.contains(&"test.cot"));
}

#[test]
fn test_manifest_operations() {
    let mut manifest = ModuleManifest::new();
    assert!(manifest.is_empty());

    manifest.add_module(
        "test.module",
        ModuleEntry::new("test.json").with_tags(vec!["test".to_string()]),
    );

    assert!(!manifest.is_empty());
    assert_eq!(manifest.len(), 1);
    assert!(manifest.contains("test.module"));

    let removed = manifest.remove("test.module");
    assert!(removed.is_some());
    assert!(manifest.is_empty());
}

#[test]
fn test_optimized_module_builder() {
    let module = OptimizedModule::new("my.module", "my.signature", "My instruction")
        .with_predictor_type(PredictorType::ChainOfThought)
        .with_tools_enabled()
        .with_demos(vec![Demo::new(
            [("input".to_string(), json!("test"))].into_iter().collect(),
            [("output".to_string(), json!("result"))]
                .into_iter()
                .collect(),
        )]);

    assert_eq!(module.module_id, "my.module");
    assert_eq!(module.signature_name, "my.signature");
    assert!(module.is_chain_of_thought());
    assert!(module.tool_enabled);
    assert_eq!(module.demo_count(), 1);
}

#[test]
fn test_error_types() {
    let err = DSPyEngineError::module_not_found("test");
    assert!(err.to_string().contains("test"));
    assert!(matches!(err.kind(), ml_crate_dsrs::inference::error::ErrorKind::Module));

    let err = DSPyEngineError::signature_not_found("sig");
    assert!(err.to_string().contains("sig"));
    assert!(matches!(err.kind(), ml_crate_dsrs::inference::error::ErrorKind::Signature));

    let err = DSPyEngineError::MaxIterationsReached(5);
    assert!(err.is_recoverable());
}

// Note: The following tests require a real model.
// Run with: cargo test --test dspy_engine_tests -- --ignored
// Ensure the Qwen2.5-0.5B model is in ./models directory

/// Helper to create a real adapter for integration tests
async fn create_test_adapter() -> Arc<CandleAdapter> {
    let pool = ModelPool::new("./models".into());
    let loaded = pool
        .load_model("Qwen2.5-0.5B")
        .await
        .expect("Failed to load model - ensure Qwen2.5-0.5B is in models/ directory");
    Arc::new(CandleAdapter::from_loaded_model(
        loaded, // load_model already returns Arc<LoadedModel>
        CandleConfig::default(),
    ))
}

/// Helper to create a test registry with the TestQASignature
fn create_test_registry() -> Arc<SignatureRegistry> {
    let mut registry = SignatureRegistry::new();
    registry.register::<TestQASignature>("test.qa");
    Arc::new(registry)
}

#[tokio::test]
#[ignore = "Requires real model in ./models/Qwen2.5-0.5B"]
async fn test_engine_load_modules() {
    let adapter = create_test_adapter().await;
    let registry = create_test_registry();

    let engine = DSPyEngine::new(fixtures_dir(), adapter, registry)
        .await
        .expect("Failed to create engine");

    // Verify modules are loaded
    assert!(engine.has_module("test.predict").await);
    assert!(engine.has_module("test.cot").await);
    assert!(engine.has_module("qa.simple").await);
    assert_eq!(engine.module_count().await, 3);

    // Verify module IDs
    let ids = engine.module_ids().await;
    assert!(ids.contains(&"test.predict".to_string()));
    assert!(ids.contains(&"test.cot".to_string()));
}

#[tokio::test]
#[ignore = "Requires real model in ./models/Qwen2.5-0.5B"]
async fn test_engine_invoke_predict() {
    let adapter = create_test_adapter().await;
    let registry = create_test_registry();

    let mut engine = DSPyEngine::new(fixtures_dir(), adapter, registry)
        .await
        .expect("Failed to create engine");

    // Configure dspy-rs before invoking
    engine.configure_dspy().await.expect("Failed to configure dspy");

    // Invoke the predict module
    let result = engine
        .invoke(
            "test.predict",
            json!({
                "question": "What is 2+2?"
            }),
        )
        .await;

    // Should succeed (may have any answer content from model)
    assert!(result.is_ok(), "Invoke failed: {:?}", result.err());
    let output = result.unwrap();
    assert!(output.is_object());

    // The result should contain an "answer" field (might be any text)
    println!("Predict output: {:?}", output);
}

#[tokio::test]
#[ignore = "Requires real model in ./models/Qwen2.5-0.5B"]
async fn test_engine_invoke_cot() {
    let adapter = create_test_adapter().await;
    let registry = create_test_registry();

    let mut engine = DSPyEngine::new(fixtures_dir(), adapter, registry)
        .await
        .expect("Failed to create engine");

    // Configure dspy-rs before invoking
    engine.configure_dspy().await.expect("Failed to configure dspy");

    // Invoke the chain-of-thought module
    let result = engine
        .invoke(
            "test.cot",
            json!({
                "question": "If I have 5 apples and give away 2, how many do I have?"
            }),
        )
        .await;

    // Should succeed
    assert!(result.is_ok(), "Invoke failed: {:?}", result.err());
    let output = result.unwrap();
    assert!(output.is_object());

    // Print output for debugging
    println!("CoT output: {:?}", output);
}

#[tokio::test]
#[ignore = "Requires real model in ./models/Qwen2.5-0.5B"]
async fn test_engine_module_not_found() {
    let adapter = create_test_adapter().await;
    let registry = create_test_registry();

    let engine = DSPyEngine::new(fixtures_dir(), adapter, registry)
        .await
        .expect("Failed to create engine");

    // Try to invoke a non-existent module
    let result = engine
        .invoke("nonexistent.module", json!({"input": "test"}))
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, DSPyEngineError::ModuleNotFound(_)));
}

#[tokio::test]
#[ignore = "Requires real model in ./models/Qwen2.5-0.5B"]
async fn test_engine_signature_not_found() {
    let adapter = create_test_adapter().await;
    // Empty registry - no signatures registered
    let registry = Arc::new(SignatureRegistry::new());

    let mut engine = DSPyEngine::new(fixtures_dir(), adapter, registry)
        .await
        .expect("Failed to create engine");

    engine.configure_dspy().await.expect("Failed to configure dspy");

    // Try to invoke a module whose signature isn't registered
    let result = engine
        .invoke("test.predict", json!({"question": "test"}))
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, DSPyEngineError::SignatureNotFound(_)));
}

#[tokio::test]
#[ignore = "Requires real model in ./models/Qwen2.5-0.5B"]
async fn test_engine_reload_module() {
    let adapter = create_test_adapter().await;
    let registry = create_test_registry();

    let engine = DSPyEngine::new(fixtures_dir(), adapter, registry)
        .await
        .expect("Failed to create engine");

    // Get initial module
    let module_before = engine.get_module("test.predict").await;
    assert!(module_before.is_some());

    // Reload the module
    engine
        .reload_module("test.predict")
        .await
        .expect("Failed to reload module");

    // Get module after reload
    let module_after = engine.get_module("test.predict").await;
    assert!(module_after.is_some());

    // Module should still have same content
    assert_eq!(
        module_before.unwrap().module_id,
        module_after.unwrap().module_id
    );
}
