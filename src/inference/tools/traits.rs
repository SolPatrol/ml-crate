//! Tool trait definition
//!
//! Defines the core `Tool` trait that all tools must implement.

use async_trait::async_trait;
use serde_json::Value;

use super::error::ToolError;

/// Abstracted tool trait for external functions the LLM can call
///
/// Tools allow DSPy modules to request external data or actions during inference.
/// This enables ReAct-style reasoning where the model can gather information
/// before generating a final response.
///
/// # Example
///
/// ```rust,ignore
/// use async_trait::async_trait;
/// use serde_json::{json, Value};
/// use ml_crate_dsrs::inference::tools::{Tool, ToolError};
///
/// struct GetPlayerGold {
///     player_id: u64,
/// }
///
/// #[async_trait]
/// impl Tool for GetPlayerGold {
///     fn name(&self) -> &str {
///         "get_player_gold"
///     }
///
///     fn description(&self) -> &str {
///         "Get the current gold amount for a player"
///     }
///
///     fn args_schema(&self) -> Option<Value> {
///         Some(json!({
///             "type": "object",
///             "properties": {},
///             "required": []
///         }))
///     }
///
///     async fn execute(&self, _args: Value) -> Result<Value, ToolError> {
///         // In a real implementation, this would query the game state
///         Ok(json!({ "gold": 500 }))
///     }
/// }
/// ```
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name (must match what the LLM outputs in tool_call)
    fn name(&self) -> &str;

    /// Human-readable description of what the tool does
    ///
    /// This is included in the `available_tools` list provided to the LLM.
    fn description(&self) -> &str;

    /// JSON schema for the tool's arguments (optional)
    ///
    /// If provided, this can be used for validation before execution.
    fn args_schema(&self) -> Option<Value> {
        None
    }

    /// Execute the tool with the given arguments
    ///
    /// # Arguments
    ///
    /// * `args` - JSON value containing the tool arguments
    ///
    /// # Returns
    ///
    /// JSON value containing the tool result, or an error.
    async fn execute(&self, args: Value) -> Result<Value, ToolError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A mock tool for testing
    struct MockTool {
        name: String,
        description: String,
        result: Value,
    }

    impl MockTool {
        fn new(name: &str, description: &str, result: Value) -> Self {
            Self {
                name: name.to_string(),
                description: description.to_string(),
                result,
            }
        }
    }

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            &self.description
        }

        async fn execute(&self, _args: Value) -> Result<Value, ToolError> {
            Ok(self.result.clone())
        }
    }

    /// A mock tool with args schema
    struct MockToolWithSchema {
        name: String,
        description: String,
        schema: Value,
    }

    impl MockToolWithSchema {
        fn new(name: &str, description: &str, schema: Value) -> Self {
            Self {
                name: name.to_string(),
                description: description.to_string(),
                schema,
            }
        }
    }

    #[async_trait]
    impl Tool for MockToolWithSchema {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            &self.description
        }

        fn args_schema(&self) -> Option<Value> {
            Some(self.schema.clone())
        }

        async fn execute(&self, args: Value) -> Result<Value, ToolError> {
            // Echo the args back
            Ok(json!({ "received_args": args }))
        }
    }

    #[tokio::test]
    async fn test_mock_tool_implements_trait() {
        let tool = MockTool::new("test_tool", "A test tool", json!({ "success": true }));

        assert_eq!(tool.name(), "test_tool");
        assert_eq!(tool.description(), "A test tool");
        assert!(tool.args_schema().is_none());

        let result = tool.execute(json!({})).await.unwrap();
        assert_eq!(result, json!({ "success": true }));
    }

    #[tokio::test]
    async fn test_tool_with_args_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "player_id": { "type": "integer" }
            },
            "required": ["player_id"]
        });

        let tool = MockToolWithSchema::new("get_player", "Get player data", schema.clone());

        assert_eq!(tool.name(), "get_player");
        assert_eq!(tool.args_schema(), Some(schema));

        let args = json!({ "player_id": 123 });
        let result = tool.execute(args.clone()).await.unwrap();
        assert_eq!(result, json!({ "received_args": args }));
    }

    #[tokio::test]
    async fn test_tool_execute_with_empty_args() {
        let tool = MockTool::new("empty_args_tool", "Takes no args", json!("done"));

        let result = tool.execute(json!({})).await.unwrap();
        assert_eq!(result, json!("done"));
    }

    #[tokio::test]
    async fn test_tool_execute_with_complex_result() {
        let complex_result = json!({
            "items": [
                { "name": "sword", "damage": 10 },
                { "name": "shield", "defense": 5 }
            ],
            "total_weight": 15.5
        });

        let tool = MockTool::new("get_inventory", "Get player inventory", complex_result.clone());

        let result = tool.execute(json!({})).await.unwrap();
        assert_eq!(result, complex_result);
    }
}
