//! RhaiTool - Wraps Rhai function pointers as the Tool trait
//!
//! Enables registering Rhai script functions as tools that can be invoked
//! by the DSPyEngine during tool-enabled module execution.
//!
//! # Architecture Note
//!
//! Rhai's `Engine` is `!Send + !Sync`, so we cannot store a reference to it.
//! Instead, RhaiTool stores:
//! - The function name (String)
//! - The AST containing the function definition (`Arc<AST>`)
//!
//! A new Engine is created for each `execute()` call.

use std::sync::Arc;

use async_trait::async_trait;
use rhai::{Engine, FnPtr, AST};
use serde_json::Value;

use crate::inference::tools::{Tool, ToolError};

use super::conversion::{dynamic_to_json, json_to_dynamic};
use super::error::RhaiConversionError;

/// A tool that wraps a Rhai function
///
/// RhaiTool allows Rhai scripts to register functions as tools that can be
/// called by DSPy modules. Since Rhai Engine is `!Send + !Sync`, we store
/// the function name and AST, creating a new Engine for each execution.
///
/// # Example
///
/// ```rust,ignore
/// // In Rhai script:
/// fn get_player_gold(args) {
///     // Tool implementation
///     #{ gold: 500 }
/// }
///
/// // Register the tool
/// dspy.register_tool("get_player_gold", "Get player's gold amount", Fn("get_player_gold"));
/// ```
pub struct RhaiTool {
    /// Tool name (must match LLM output in tool_call)
    name: String,

    /// Human-readable description
    description: String,

    /// The name of the function to call in the AST
    fn_name: String,

    /// The AST containing the function definition
    ast: Arc<AST>,
}

impl RhaiTool {
    /// Create a new RhaiTool
    ///
    /// # Arguments
    ///
    /// * `name` - The tool name (used in LLM tool_call)
    /// * `description` - Human-readable description
    /// * `fn_ptr` - Rhai function pointer to the tool implementation
    /// * `ast` - The AST containing the function definition
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tool = RhaiTool::new(
    ///     "get_gold",
    ///     "Get player gold",
    ///     fn_ptr,
    ///     Arc::new(ast),
    /// );
    /// ```
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        fn_ptr: FnPtr,
        ast: Arc<AST>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            fn_name: fn_ptr.fn_name().to_string(),
            ast,
        }
    }

    /// Create a new RhaiTool with explicit function name
    ///
    /// Use this when you have the function name as a string rather than a FnPtr.
    pub fn with_fn_name(
        name: impl Into<String>,
        description: impl Into<String>,
        fn_name: impl Into<String>,
        ast: Arc<AST>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            fn_name: fn_name.into(),
            ast,
        }
    }

    /// Get the function name being called
    pub fn fn_name(&self) -> &str {
        &self.fn_name
    }

    /// Execute the Rhai function with the given arguments
    ///
    /// This creates a new Engine for the call since Engine is !Send + !Sync.
    fn execute_sync(&self, args: Value) -> Result<Value, ToolError> {
        // Create a new engine for this call
        let engine = Engine::new();

        // Convert JSON args to Rhai Dynamic
        let dynamic_args = json_to_dynamic(args);

        // Call the function
        let result = engine
            .call_fn::<rhai::Dynamic>(&mut rhai::Scope::new(), &self.ast, &self.fn_name, (dynamic_args,))
            .map_err(|e| ToolError::execution_failed(format!("Rhai execution error: {}", e)))?;

        // Convert result back to JSON
        dynamic_to_json(result).map_err(|e| match e {
            RhaiConversionError::UnsupportedType(t) => {
                ToolError::execution_failed(format!("Cannot convert Rhai result to JSON: unsupported type {}", t))
            }
            RhaiConversionError::NonStringKey => {
                ToolError::execution_failed("Cannot convert Rhai result to JSON: map has non-string keys")
            }
        })
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

    fn args_schema(&self) -> Option<Value> {
        // Rhai functions don't have compile-time schemas
        None
    }

    async fn execute(&self, args: Value) -> Result<Value, ToolError> {
        // Execute synchronously - Rhai Engine is !Send
        // This is fine since Rhai execution is typically fast
        self.execute_sync(args)
    }
}

// Safety: RhaiTool is Send + Sync because:
// - name, description, fn_name are String (Send + Sync)
// - ast is Arc<AST> where AST is Send + Sync
// We don't store the Engine (which is !Send + !Sync)
unsafe impl Send for RhaiTool {}
unsafe impl Sync for RhaiTool {}

impl std::fmt::Debug for RhaiTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RhaiTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("fn_name", &self.fn_name)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn create_test_ast(script: &str) -> Arc<AST> {
        let engine = Engine::new();
        let ast = engine.compile(script).expect("Failed to compile test script");
        Arc::new(ast)
    }

    #[test]
    fn test_rhai_tool_creation() {
        let ast = create_test_ast(
            r#"
            fn test_tool(args) {
                42
            }
            "#,
        );

        let engine = Engine::new();
        let fn_ptr = engine.eval::<FnPtr>(r#"Fn("test_tool")"#).unwrap();

        let tool = RhaiTool::new("test_tool", "A test tool", fn_ptr, ast);

        assert_eq!(tool.name(), "test_tool");
        assert_eq!(tool.description(), "A test tool");
        assert_eq!(tool.fn_name(), "test_tool");
        assert!(tool.args_schema().is_none());
    }

    #[test]
    fn test_rhai_tool_with_fn_name() {
        let ast = create_test_ast(
            r#"
            fn my_function(args) {
                "hello"
            }
            "#,
        );

        let tool = RhaiTool::with_fn_name("my_tool", "My tool", "my_function", ast);

        assert_eq!(tool.name(), "my_tool");
        assert_eq!(tool.fn_name(), "my_function");
    }

    #[tokio::test]
    async fn test_rhai_tool_execute_simple() {
        let ast = create_test_ast(
            r#"
            fn simple_tool(args) {
                42
            }
            "#,
        );

        let tool = RhaiTool::with_fn_name("simple", "Returns 42", "simple_tool", ast);

        let result = tool.execute(json!({})).await.unwrap();
        assert_eq!(result, json!(42));
    }

    #[tokio::test]
    async fn test_rhai_tool_execute_with_args() {
        let ast = create_test_ast(
            r#"
            fn add_tool(args) {
                args.a + args.b
            }
            "#,
        );

        let tool = RhaiTool::with_fn_name("add", "Adds two numbers", "add_tool", ast);

        let result = tool.execute(json!({"a": 10, "b": 32})).await.unwrap();
        assert_eq!(result, json!(42));
    }

    #[tokio::test]
    async fn test_rhai_tool_execute_returns_map() {
        let ast = create_test_ast(
            r#"
            fn map_tool(args) {
                #{
                    gold: 500,
                    items: ["sword", "shield"]
                }
            }
            "#,
        );

        let tool = RhaiTool::with_fn_name("inventory", "Get inventory", "map_tool", ast);

        let result = tool.execute(json!({})).await.unwrap();
        assert_eq!(result["gold"], 500);
        assert_eq!(result["items"][0], "sword");
        assert_eq!(result["items"][1], "shield");
    }

    #[tokio::test]
    async fn test_rhai_tool_execute_returns_array() {
        let ast = create_test_ast(
            r#"
            fn array_tool(args) {
                [1, 2, 3, 4, 5]
            }
            "#,
        );

        let tool = RhaiTool::with_fn_name("numbers", "Get numbers", "array_tool", ast);

        let result = tool.execute(json!({})).await.unwrap();
        assert_eq!(result, json!([1, 2, 3, 4, 5]));
    }

    #[tokio::test]
    async fn test_rhai_tool_error_handling() {
        let ast = create_test_ast(
            r#"
            fn error_tool(args) {
                throw "Something went wrong!";
            }
            "#,
        );

        let tool = RhaiTool::with_fn_name("error", "Throws an error", "error_tool", ast);

        let result = tool.execute(json!({})).await;
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed(_)));
        assert!(err.to_string().contains("Something went wrong"));
    }

    #[tokio::test]
    async fn test_rhai_tool_function_not_found() {
        let ast = create_test_ast(
            r#"
            fn other_function(args) {
                42
            }
            "#,
        );

        let tool = RhaiTool::with_fn_name("missing", "Calls missing function", "nonexistent", ast);

        let result = tool.execute(json!({})).await;
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed(_)));
    }

    #[tokio::test]
    async fn test_rhai_tool_complex_args() {
        let ast = create_test_ast(
            r#"
            fn complex_tool(args) {
                let sum = 0;
                for item in args.items {
                    sum += item.value;
                }
                #{
                    total: sum,
                    count: args.items.len()
                }
            }
            "#,
        );

        let tool = RhaiTool::with_fn_name("complex", "Process complex args", "complex_tool", ast);

        let result = tool
            .execute(json!({
                "items": [
                    {"value": 10},
                    {"value": 20},
                    {"value": 30}
                ]
            }))
            .await
            .unwrap();

        assert_eq!(result["total"], 60);
        assert_eq!(result["count"], 3);
    }

    #[test]
    fn test_rhai_tool_debug() {
        let ast = create_test_ast("fn test(args) { 0 }");
        let tool = RhaiTool::with_fn_name("test", "Test tool", "test", ast);

        let debug_str = format!("{:?}", tool);
        assert!(debug_str.contains("RhaiTool"));
        assert!(debug_str.contains("test"));
    }
}
