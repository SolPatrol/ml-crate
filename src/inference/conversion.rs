//! Value ↔ Example conversion helpers
//!
//! Provides utilities for converting between serde_json::Value and dspy-rs
//! Example/Prediction types. These conversions are essential for the DSPyEngine
//! which works with JSON inputs but needs to interface with dspy-rs types.
//!
//! # Example
//!
//! ```rust,ignore
//! use serde_json::json;
//! use ml_crate_dsrs::inference::conversion::{value_to_example, prediction_to_value};
//!
//! // Convert JSON to Example
//! let input = json!({"question": "What is 2+2?"});
//! let example = value_to_example(&input, &["question"], &[]);
//!
//! // Convert Prediction back to JSON
//! let output = prediction_to_value(&prediction);
//! ```

use dspy_rs::{Example, LmUsage, MetaSignature, Prediction};
use serde_json::Value;
use std::collections::HashMap;

use super::module::Demo;

/// Convert a serde_json::Value to a dspy-rs Example
///
/// Creates an Example from a JSON object, using the signature to determine
/// which fields are inputs and which are outputs.
///
/// # Arguments
///
/// * `value` - JSON object containing field values
/// * `signature` - MetaSignature providing input/output field names
///
/// # Returns
///
/// An Example with properly classified input and output keys.
///
/// # Example
///
/// ```rust,ignore
/// use serde_json::json;
///
/// let value = json!({
///     "question": "What is Rust?",
///     "context": "Programming languages"
/// });
///
/// let example = value_to_example(&value, signature);
/// ```
pub fn value_to_example(value: &Value, signature: &dyn MetaSignature) -> Example {
    let input_fields = signature.input_fields();
    let output_fields = signature.output_fields();

    // Extract input and output key names from signature
    let input_keys = extract_field_names(&input_fields);
    let output_keys = extract_field_names(&output_fields);

    value_to_example_with_keys(value, &input_keys, &output_keys)
}

/// Convert a Value to Example with explicit input/output keys
///
/// This is useful when you know the field names without needing a signature.
///
/// # Arguments
///
/// * `value` - JSON object containing field values
/// * `input_keys` - Names of input fields
/// * `output_keys` - Names of output fields
pub fn value_to_example_with_keys(
    value: &Value,
    input_keys: &[String],
    output_keys: &[String],
) -> Example {
    let data: HashMap<String, Value> = match value.as_object() {
        Some(obj) => obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        None => HashMap::new(),
    };

    Example::new(data, input_keys.to_vec(), output_keys.to_vec())
}

/// Convert a Demo to a dspy-rs Example
///
/// Demos from module JSON files need to be converted to Examples
/// for use with dspy-rs predictors.
pub fn demo_to_example(demo: &Demo) -> Example {
    let mut data = HashMap::new();

    // Add all inputs
    for (key, value) in &demo.inputs {
        data.insert(key.clone(), value.clone());
    }

    // Add all outputs
    for (key, value) in &demo.outputs {
        data.insert(key.clone(), value.clone());
    }

    let input_keys: Vec<String> = demo.inputs.keys().cloned().collect();
    let output_keys: Vec<String> = demo.outputs.keys().cloned().collect();

    Example::new(data, input_keys, output_keys)
}

/// Convert multiple Demos to Examples
pub fn demos_to_examples(demos: &[Demo]) -> Vec<Example> {
    demos.iter().map(demo_to_example).collect()
}

/// Convert a dspy-rs Example to a serde_json::Value
///
/// Useful for serializing Example data to JSON.
pub fn example_to_value(example: &Example) -> Value {
    serde_json::to_value(&example.data).unwrap_or(Value::Null)
}

/// Extract only input fields from an Example as a Value
pub fn example_inputs_to_value(example: &Example) -> Value {
    let inputs: HashMap<String, Value> = example
        .input_keys
        .iter()
        .filter_map(|k| example.data.get(k).map(|v| (k.clone(), v.clone())))
        .collect();

    serde_json::to_value(inputs).unwrap_or(Value::Null)
}

/// Extract only output fields from an Example as a Value
pub fn example_outputs_to_value(example: &Example) -> Value {
    let outputs: HashMap<String, Value> = example
        .output_keys
        .iter()
        .filter_map(|k| example.data.get(k).map(|v| (k.clone(), v.clone())))
        .collect();

    serde_json::to_value(outputs).unwrap_or(Value::Null)
}

/// Convert a dspy-rs Prediction to a serde_json::Value
///
/// # Arguments
///
/// * `prediction` - The prediction to convert
///
/// # Returns
///
/// A JSON object containing the prediction data.
pub fn prediction_to_value(prediction: &Prediction) -> Value {
    serde_json::to_value(&prediction.data).unwrap_or(Value::Null)
}

/// Convert a Prediction to Value with usage metadata
///
/// Returns a JSON object that includes both the prediction data
/// and token usage information.
pub fn prediction_to_value_with_usage(prediction: &Prediction) -> Value {
    serde_json::json!({
        "data": prediction.data,
        "usage": {
            "prompt_tokens": prediction.lm_usage.prompt_tokens,
            "completion_tokens": prediction.lm_usage.completion_tokens,
            "total_tokens": prediction.lm_usage.total_tokens
        }
    })
}

/// Create a Prediction from a JSON Value
///
/// Useful for testing or creating predictions programmatically.
pub fn value_to_prediction(value: &Value) -> Prediction {
    let data: HashMap<String, Value> = match value.as_object() {
        Some(obj) => obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        None => HashMap::new(),
    };

    Prediction::new(data, LmUsage::default())
}

/// Create a Prediction from a JSON Value with usage stats
pub fn value_to_prediction_with_usage(
    value: &Value,
    prompt_tokens: u64,
    completion_tokens: u64,
) -> Prediction {
    let data: HashMap<String, Value> = match value.as_object() {
        Some(obj) => obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        None => HashMap::new(),
    };

    Prediction::new(
        data,
        LmUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    )
}

/// Extract field names from a signature's input_fields() or output_fields() Value
///
/// The signature returns field info as JSON, typically as an array of strings
/// or an array of objects with "name" keys.
fn extract_field_names(fields_value: &Value) -> Vec<String> {
    match fields_value.as_array() {
        Some(arr) => arr
            .iter()
            .filter_map(|v| {
                // Handle both string and object formats
                if let Some(s) = v.as_str() {
                    Some(s.to_string())
                } else if let Some(obj) = v.as_object() {
                    obj.get("name").and_then(|n| n.as_str()).map(String::from)
                } else {
                    None
                }
            })
            .collect(),
        None => Vec::new(),
    }
}

/// Merge two JSON objects
///
/// Useful for combining inputs with tool results.
pub fn merge_values(base: &Value, overlay: &Value) -> Value {
    match (base, overlay) {
        (Value::Object(base_map), Value::Object(overlay_map)) => {
            let mut result = base_map.clone();
            for (k, v) in overlay_map {
                result.insert(k.clone(), v.clone());
            }
            Value::Object(result)
        }
        _ => overlay.clone(),
    }
}

/// Get a string value from a JSON object
pub fn get_string(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(|v| v.as_str()).map(String::from)
}

/// Get a string value with a default
pub fn get_string_or(value: &Value, key: &str, default: &str) -> String {
    get_string(value, key).unwrap_or_else(|| default.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Mock signature for testing
    struct MockSignature;

    impl MetaSignature for MockSignature {
        fn demos(&self) -> Vec<Example> {
            vec![]
        }

        fn set_demos(&mut self, _demos: Vec<Example>) -> anyhow::Result<()> {
            Ok(())
        }

        fn instruction(&self) -> String {
            "Test".to_string()
        }

        fn input_fields(&self) -> Value {
            json!(["question", "context"])
        }

        fn output_fields(&self) -> Value {
            json!(["answer"])
        }

        fn update_instruction(&mut self, _instruction: String) -> anyhow::Result<()> {
            Ok(())
        }

        fn append(&mut self, _name: &str, _value: Value) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_value_to_example_basic() {
        let value = json!({
            "question": "What is 2+2?",
            "context": "Math"
        });

        let signature = MockSignature;
        let example = value_to_example(&value, &signature);

        assert!(example.input_keys.contains(&"question".to_string()));
        assert!(example.input_keys.contains(&"context".to_string()));
        assert!(example.output_keys.contains(&"answer".to_string()));
    }

    #[test]
    fn test_value_to_example_with_keys() {
        let value = json!({
            "input1": "value1",
            "input2": "value2",
            "output1": "result"
        });

        let example = value_to_example_with_keys(
            &value,
            &["input1".to_string(), "input2".to_string()],
            &["output1".to_string()],
        );

        assert_eq!(example.input_keys.len(), 2);
        assert_eq!(example.output_keys.len(), 1);
        assert_eq!(
            example.data.get("input1").unwrap().as_str().unwrap(),
            "value1"
        );
    }

    #[test]
    fn test_demo_to_example() {
        let demo = Demo::new(
            [("question".to_string(), json!("What is Rust?"))]
                .into_iter()
                .collect(),
            [("answer".to_string(), json!("A programming language"))]
                .into_iter()
                .collect(),
        );

        let example = demo_to_example(&demo);

        assert!(example.input_keys.contains(&"question".to_string()));
        assert!(example.output_keys.contains(&"answer".to_string()));
        assert_eq!(
            example.data.get("question").unwrap().as_str().unwrap(),
            "What is Rust?"
        );
    }

    #[test]
    fn test_demos_to_examples() {
        let demos = vec![
            Demo::new(
                [("q".to_string(), json!("Q1"))].into_iter().collect(),
                [("a".to_string(), json!("A1"))].into_iter().collect(),
            ),
            Demo::new(
                [("q".to_string(), json!("Q2"))].into_iter().collect(),
                [("a".to_string(), json!("A2"))].into_iter().collect(),
            ),
        ];

        let examples = demos_to_examples(&demos);

        assert_eq!(examples.len(), 2);
    }

    #[test]
    fn test_example_to_value() {
        let data: HashMap<String, Value> = [
            ("question".to_string(), json!("What?")),
            ("answer".to_string(), json!("Something")),
        ]
        .into_iter()
        .collect();

        let example = Example::new(
            data,
            vec!["question".to_string()],
            vec!["answer".to_string()],
        );

        let value = example_to_value(&example);

        assert!(value.is_object());
        assert_eq!(value.get("question").unwrap().as_str().unwrap(), "What?");
    }

    #[test]
    fn test_example_inputs_to_value() {
        let data: HashMap<String, Value> = [
            ("input".to_string(), json!("in")),
            ("output".to_string(), json!("out")),
        ]
        .into_iter()
        .collect();

        let example = Example::new(
            data,
            vec!["input".to_string()],
            vec!["output".to_string()],
        );

        let inputs = example_inputs_to_value(&example);

        assert!(inputs.is_object());
        assert!(inputs.get("input").is_some());
        assert!(inputs.get("output").is_none());
    }

    #[test]
    fn test_prediction_to_value() {
        let data: HashMap<String, Value> =
            [("answer".to_string(), json!("42"))].into_iter().collect();

        let prediction = Prediction::new(data, LmUsage::default());

        let value = prediction_to_value(&prediction);

        assert_eq!(value.get("answer").unwrap().as_str().unwrap(), "42");
    }

    #[test]
    fn test_prediction_to_value_with_usage() {
        let data: HashMap<String, Value> =
            [("answer".to_string(), json!("42"))].into_iter().collect();

        let prediction = Prediction::new(data, LmUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        });

        let value = prediction_to_value_with_usage(&prediction);

        assert!(value.get("data").is_some());
        assert!(value.get("usage").is_some());
        assert_eq!(value["usage"]["prompt_tokens"], 10);
        assert_eq!(value["usage"]["completion_tokens"], 5);
        assert_eq!(value["usage"]["total_tokens"], 15);
    }

    #[test]
    fn test_value_to_prediction() {
        let value = json!({
            "answer": "42",
            "confidence": 0.95
        });

        let prediction = value_to_prediction(&value);

        assert_eq!(
            prediction.data.get("answer").unwrap().as_str().unwrap(),
            "42"
        );
        assert_eq!(prediction.lm_usage.total_tokens, 0);
    }

    #[test]
    fn test_value_to_prediction_with_usage() {
        let value = json!({"answer": "42"});

        let prediction = value_to_prediction_with_usage(&value, 100, 50);

        assert_eq!(prediction.lm_usage.prompt_tokens, 100);
        assert_eq!(prediction.lm_usage.completion_tokens, 50);
        assert_eq!(prediction.lm_usage.total_tokens, 150);
    }

    #[test]
    fn test_extract_field_names_from_string_array() {
        let fields = json!(["field1", "field2", "field3"]);
        let names = extract_field_names(&fields);

        assert_eq!(names, vec!["field1", "field2", "field3"]);
    }

    #[test]
    fn test_extract_field_names_from_object_array() {
        let fields = json!([
            {"name": "field1", "description": "First field"},
            {"name": "field2", "description": "Second field"}
        ]);
        let names = extract_field_names(&fields);

        assert_eq!(names, vec!["field1", "field2"]);
    }

    #[test]
    fn test_merge_values() {
        let base = json!({
            "a": 1,
            "b": 2
        });
        let overlay = json!({
            "b": 3,
            "c": 4
        });

        let merged = merge_values(&base, &overlay);

        assert_eq!(merged["a"], 1);
        assert_eq!(merged["b"], 3); // Overlay wins
        assert_eq!(merged["c"], 4);
    }

    #[test]
    fn test_get_string() {
        let value = json!({
            "name": "test",
            "count": 42
        });

        assert_eq!(get_string(&value, "name"), Some("test".to_string()));
        assert_eq!(get_string(&value, "count"), None); // Not a string
        assert_eq!(get_string(&value, "missing"), None);
    }

    #[test]
    fn test_get_string_or() {
        let value = json!({"name": "test"});

        assert_eq!(get_string_or(&value, "name", "default"), "test");
        assert_eq!(get_string_or(&value, "missing", "default"), "default");
    }

    #[test]
    fn test_empty_value_to_example() {
        let value = json!({});
        let example = value_to_example_with_keys(&value, &[], &[]);

        assert!(example.data.is_empty());
        assert!(example.input_keys.is_empty());
        assert!(example.output_keys.is_empty());
    }

    #[test]
    fn test_non_object_value() {
        let value = json!("just a string");
        let example = value_to_example_with_keys(
            &value,
            &["input".to_string()],
            &["output".to_string()],
        );

        // Should handle gracefully with empty data
        assert!(example.data.is_empty());
    }
}
