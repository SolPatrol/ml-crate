//! Tool System for DSPy Engine
//!
//! This module provides the tool infrastructure that allows DSPy modules to
//! request and execute external tools (game functions, data lookups, etc.)
//! during inference. This enables ReAct-style reasoning where the LLM can
//! gather information before responding.
//!
//! # Architecture
//!
//! ```text
//! LLM Output
//!     ↓
//! { "tool_call": { "name": "get_gold", "args": {} } }
//!     ↓
//! ToolWrapper parses tool_call
//!     ↓
//! ToolRegistry.execute("get_gold", args)
//!     ↓
//! Tool.execute() → Result
//!     ↓
//! Inject result into context
//!     ↓
//! Re-invoke predictor
//!     ↓
//! Repeat until no tool_call or max iterations
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use ml_crate_dsrs::inference::tools::{Tool, ToolRegistry, ToolCall};
//! use async_trait::async_trait;
//! use serde_json::json;
//!
//! // Define a tool
//! struct GetPlayerGold;
//!
//! #[async_trait]
//! impl Tool for GetPlayerGold {
//!     fn name(&self) -> &str { "get_player_gold" }
//!     fn description(&self) -> &str { "Get player's gold" }
//!     async fn execute(&self, _args: Value) -> Result<Value, ToolError> {
//!         Ok(json!({ "gold": 500 }))
//!     }
//! }
//!
//! // Register tool
//! let mut registry = ToolRegistry::new();
//! registry.register(Arc::new(GetPlayerGold));
//!
//! // Execute tool call
//! let call = ToolCall::new("get_player_gold", json!({}));
//! let result = registry.execute_call(&call).await?;
//! ```

pub mod error;
pub mod registry;
pub mod traits;
pub mod wrapper;

// Re-export commonly used types
pub use error::{ToolError, ToolErrorKind};
pub use registry::ToolRegistry;
pub use traits::Tool;
pub use wrapper::{ToolWrapper, ToolWrapperConfig};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A tool call request parsed from LLM output
///
/// This represents a request from the LLM to execute a specific tool
/// with the provided arguments.
///
/// # Example JSON
///
/// ```json
/// {
///   "name": "get_player_gold",
///   "args": { "player_id": 123 }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// Name of the tool to call
    pub name: String,
    /// Arguments to pass to the tool
    #[serde(default)]
    pub args: Value,
}

impl ToolCall {
    /// Create a new tool call
    pub fn new(name: impl Into<String>, args: Value) -> Self {
        Self {
            name: name.into(),
            args,
        }
    }

    /// Create a tool call with no arguments
    pub fn no_args(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            args: Value::Object(Default::default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tool_call_deserialize_from_json() {
        let json = r#"{"name": "get_player_gold", "args": {"player_id": 123}}"#;
        let call: ToolCall = serde_json::from_str(json).unwrap();

        assert_eq!(call.name, "get_player_gold");
        assert_eq!(call.args, json!({"player_id": 123}));
    }

    #[test]
    fn test_tool_call_serialize_to_json() {
        let call = ToolCall::new("get_inventory", json!({}));
        let json = serde_json::to_string(&call).unwrap();

        assert!(json.contains("\"name\":\"get_inventory\""));
        assert!(json.contains("\"args\":{}"));
    }

    #[test]
    fn test_tool_call_deserialize_without_args() {
        // args should default to empty object when missing
        let json = r#"{"name": "ping"}"#;
        let call: ToolCall = serde_json::from_str(json).unwrap();

        assert_eq!(call.name, "ping");
        assert_eq!(call.args, Value::Null); // default serde behavior
    }

    #[test]
    fn test_tool_call_no_args_constructor() {
        let call = ToolCall::no_args("simple_tool");

        assert_eq!(call.name, "simple_tool");
        assert_eq!(call.args, json!({}));
    }

    #[test]
    fn test_tool_call_with_complex_args() {
        let args = json!({
            "player_id": 123,
            "items": ["sword", "shield"],
            "options": {
                "include_equipped": true
            }
        });

        let call = ToolCall::new("get_player_items", args.clone());
        assert_eq!(call.args, args);

        // Round-trip serialization
        let json_str = serde_json::to_string(&call).unwrap();
        let deserialized: ToolCall = serde_json::from_str(&json_str).unwrap();
        assert_eq!(deserialized, call);
    }

    #[test]
    fn test_tool_call_equality() {
        let call1 = ToolCall::new("test", json!({"a": 1}));
        let call2 = ToolCall::new("test", json!({"a": 1}));
        let call3 = ToolCall::new("test", json!({"a": 2}));

        assert_eq!(call1, call2);
        assert_ne!(call1, call3);
    }
}
