//! Tool Registry
//!
//! Stores and manages registered tools, providing lookup and execution.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};

use super::error::ToolError;
use super::traits::Tool;
use super::ToolCall;

/// Registry for storing and executing tools
///
/// The ToolRegistry manages a collection of tools that can be called by
/// the LLM during inference. It provides methods for registration, lookup,
/// and execution.
///
/// # Example
///
/// ```rust,ignore
/// use ml_crate_dsrs::inference::tools::{Tool, ToolRegistry, ToolError};
/// use async_trait::async_trait;
/// use serde_json::{json, Value};
/// use std::sync::Arc;
///
/// struct MyTool;
///
/// #[async_trait]
/// impl Tool for MyTool {
///     fn name(&self) -> &str { "my_tool" }
///     fn description(&self) -> &str { "Does something useful" }
///     async fn execute(&self, _args: Value) -> Result<Value, ToolError> {
///         Ok(json!({ "result": "success" }))
///     }
/// }
///
/// let mut registry = ToolRegistry::new();
/// registry.register(Arc::new(MyTool));
///
/// let result = registry.execute("my_tool", json!({})).await?;
/// ```
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new empty tool registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool
    ///
    /// The tool will be stored under its name (from `Tool::name()`).
    /// If a tool with the same name already exists, it will be replaced.
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Check if a tool is registered
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Get the number of registered tools
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Get all registered tool names
    pub fn names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Execute a tool by name
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the tool to execute
    /// * `args` - JSON arguments to pass to the tool
    ///
    /// # Returns
    ///
    /// The tool's result as a JSON value, or an error if the tool
    /// wasn't found or execution failed.
    pub async fn execute(&self, name: &str, args: Value) -> Result<Value, ToolError> {
        let tool = self
            .get(name)
            .ok_or_else(|| ToolError::not_found(name))?;
        tool.execute(args).await
    }

    /// Execute a tool call
    ///
    /// Convenience method that extracts name and args from a ToolCall struct.
    pub async fn execute_call(&self, call: &ToolCall) -> Result<Value, ToolError> {
        self.execute(&call.name, call.args.clone()).await
    }

    /// Generate JSON list of available tools for LLM context
    ///
    /// Returns a JSON array describing all registered tools, suitable
    /// for inclusion in the `available_tools` input field.
    ///
    /// # Example Output
    ///
    /// ```json
    /// [
    ///   {
    ///     "name": "get_player_gold",
    ///     "description": "Get player's current gold amount",
    ///     "args_schema": null
    ///   },
    ///   {
    ///     "name": "get_inventory",
    ///     "description": "Get player's inventory",
    ///     "args_schema": { "type": "object", "properties": {} }
    ///   }
    /// ]
    /// ```
    pub fn to_json(&self) -> Value {
        let tools: Vec<Value> = self
            .tools
            .values()
            .map(|t| {
                json!({
                    "name": t.name(),
                    "description": t.description(),
                    "args_schema": t.args_schema()
                })
            })
            .collect();
        Value::Array(tools)
    }
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    /// Simple mock tool for testing
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

    /// Mock tool that returns its args
    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echoes back the arguments"
        }

        async fn execute(&self, args: Value) -> Result<Value, ToolError> {
            Ok(json!({ "echoed": args }))
        }
    }

    /// Mock tool that always fails
    struct FailingTool;

    #[async_trait]
    impl Tool for FailingTool {
        fn name(&self) -> &str {
            "failing_tool"
        }

        fn description(&self) -> &str {
            "Always fails"
        }

        async fn execute(&self, _args: Value) -> Result<Value, ToolError> {
            Err(ToolError::execution_failed("intentional failure"))
        }
    }

    #[test]
    fn test_registry_new() {
        let registry = ToolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_default() {
        let registry = ToolRegistry::default();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_register_and_retrieve_tool() {
        let mut registry = ToolRegistry::new();
        let tool = Arc::new(MockTool::new("test", "A test tool", json!({"ok": true})));

        registry.register(tool);

        assert!(registry.contains("test"));
        assert_eq!(registry.len(), 1);

        let retrieved = registry.get("test");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name(), "test");
    }

    #[test]
    fn test_get_nonexistent_tool() {
        let registry = ToolRegistry::new();
        assert!(registry.get("nonexistent").is_none());
        assert!(!registry.contains("nonexistent"));
    }

    #[tokio::test]
    async fn test_execute_registered_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(MockTool::new(
            "get_gold",
            "Get gold",
            json!({"gold": 500}),
        )));

        let result = registry.execute("get_gold", json!({})).await.unwrap();
        assert_eq!(result, json!({"gold": 500}));
    }

    #[tokio::test]
    async fn test_execute_call_with_tool_call_struct() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool));

        let call = ToolCall::new("echo", json!({"message": "hello"}));
        let result = registry.execute_call(&call).await.unwrap();

        assert_eq!(result, json!({"echoed": {"message": "hello"}}));
    }

    #[tokio::test]
    async fn test_execute_nonexistent_tool_returns_not_found() {
        let registry = ToolRegistry::new();

        let result = registry.execute("nonexistent", json!({})).await;
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::NotFound(_)));
        assert!(err.to_string().contains("nonexistent"));
    }

    #[tokio::test]
    async fn test_execute_failing_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(FailingTool));

        let result = registry.execute("failing_tool", json!({})).await;
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed(_)));
    }

    #[test]
    fn test_to_json_format() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(MockTool::new(
            "get_gold",
            "Get player gold",
            json!({}),
        )));
        registry.register(Arc::new(MockTool::new(
            "get_inventory",
            "Get player inventory",
            json!({}),
        )));

        let json_list = registry.to_json();

        assert!(json_list.is_array());
        let arr = json_list.as_array().unwrap();
        assert_eq!(arr.len(), 2);

        // Check that each tool has the expected fields
        for tool_json in arr {
            assert!(tool_json.get("name").is_some());
            assert!(tool_json.get("description").is_some());
            assert!(tool_json.get("args_schema").is_some());
        }
    }

    #[test]
    fn test_names() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(MockTool::new("tool1", "desc1", json!({}))));
        registry.register(Arc::new(MockTool::new("tool2", "desc2", json!({}))));

        let names = registry.names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"tool1"));
        assert!(names.contains(&"tool2"));
    }

    #[test]
    fn test_register_overwrites_existing() {
        let mut registry = ToolRegistry::new();

        registry.register(Arc::new(MockTool::new(
            "test",
            "Original",
            json!({"v": 1}),
        )));
        registry.register(Arc::new(MockTool::new(
            "test",
            "Replaced",
            json!({"v": 2}),
        )));

        assert_eq!(registry.len(), 1);

        let tool = registry.get("test").unwrap();
        assert_eq!(tool.description(), "Replaced");
    }

    #[test]
    fn test_debug_impl() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(MockTool::new("test", "desc", json!({}))));

        let debug_str = format!("{:?}", registry);
        assert!(debug_str.contains("ToolRegistry"));
        assert!(debug_str.contains("test"));
    }
}
