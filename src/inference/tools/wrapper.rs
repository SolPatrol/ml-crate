//! Tool Wrapper
//!
//! Generic wrapper that adds tool-calling capability to predictor invocations.

use std::sync::Arc;

use serde_json::Value;

use super::error::ToolError;
use super::registry::ToolRegistry;
use super::ToolCall;

/// Configuration for the tool wrapper
#[derive(Debug, Clone)]
pub struct ToolWrapperConfig {
    /// Maximum number of tool call iterations before returning an error
    pub max_iterations: usize,
    /// Key name for injecting tool results into input
    pub tool_result_key: String,
}

impl Default for ToolWrapperConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            tool_result_key: "context".to_string(),
        }
    }
}

impl ToolWrapperConfig {
    /// Create a new config with custom max iterations
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Create a new config with custom tool result key
    pub fn with_tool_result_key(mut self, key: impl Into<String>) -> Self {
        self.tool_result_key = key.into();
        self
    }
}

/// Generic tool wrapper for adding tool-calling capability to predictors
///
/// The ToolWrapper manages the tool execution loop:
/// 1. Inject available_tools into input
/// 2. Invoke the predictor
/// 3. Check output for tool_call field
/// 4. If tool_call present, execute tool and append result to context
/// 5. Repeat until no tool_call or max iterations
///
/// # Example
///
/// ```rust,ignore
/// use ml_crate_dsrs::inference::tools::{ToolWrapper, ToolWrapperConfig, ToolRegistry};
/// use std::sync::Arc;
///
/// let registry = Arc::new(ToolRegistry::new());
/// let config = ToolWrapperConfig::default();
/// let wrapper = ToolWrapper::new(registry, config);
///
/// // Use via DSPyEngine.invoke_with_tools()
/// ```
pub struct ToolWrapper {
    tools: Arc<ToolRegistry>,
    config: ToolWrapperConfig,
}

impl ToolWrapper {
    /// Create a new tool wrapper
    pub fn new(tools: Arc<ToolRegistry>, config: ToolWrapperConfig) -> Self {
        Self { tools, config }
    }

    /// Get the tool registry
    pub fn tools(&self) -> &Arc<ToolRegistry> {
        &self.tools
    }

    /// Get the configuration
    pub fn config(&self) -> &ToolWrapperConfig {
        &self.config
    }

    /// Check if a tool_call is present and not null in the output
    fn has_tool_call(output: &Value) -> bool {
        match output.get("tool_call") {
            Some(v) => !v.is_null(),
            None => false,
        }
    }

    /// Parse a tool_call from output
    fn parse_tool_call(output: &Value) -> Result<ToolCall, ToolError> {
        let tool_call_value = output
            .get("tool_call")
            .ok_or_else(|| ToolError::serialization("No tool_call field in output"))?;

        serde_json::from_value(tool_call_value.clone())
            .map_err(|e| ToolError::serialization(format!("Failed to parse tool_call: {}", e)))
    }

    /// Append tool result to the context field in input
    fn append_tool_result(&self, input: &mut Value, tool_name: &str, result: &Value) {
        let context_update = format!(
            "Tool '{}' returned: {}",
            tool_name,
            serde_json::to_string(result).unwrap_or_else(|_| "{}".to_string())
        );

        // Get existing context or empty string
        let existing = input
            .get(&self.config.tool_result_key)
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Append new context
        let new_context = if existing.is_empty() {
            context_update
        } else {
            format!("{}\n{}", existing, context_update)
        };

        input[&self.config.tool_result_key] = Value::String(new_context);
    }

    /// Invoke with tool support using a provided invocation function
    ///
    /// This is the core method that handles the tool execution loop.
    /// The `invoke_fn` parameter allows this to be called from DSPyEngine
    /// without circular dependencies.
    ///
    /// # Arguments
    ///
    /// * `invoke_fn` - Async function that performs the actual predictor invocation
    /// * `input` - Initial input to the predictor
    ///
    /// # Returns
    ///
    /// The final output after all tool calls are complete, or an error.
    pub async fn invoke_with_fn<F, Fut>(
        &self,
        invoke_fn: F,
        mut input: Value,
    ) -> Result<Value, InvokeWithToolsError>
    where
        F: Fn(Value) -> Fut,
        Fut: std::future::Future<Output = Result<Value, String>>,
    {
        // Inject available tools into input
        input["available_tools"] = self.tools.to_json();

        for iteration in 0..self.config.max_iterations {
            tracing::debug!(
                "Tool wrapper iteration {}/{}",
                iteration + 1,
                self.config.max_iterations
            );

            // Call the underlying predictor
            let output = invoke_fn(input.clone())
                .await
                .map_err(InvokeWithToolsError::InvokeError)?;

            // Check for tool call in output
            if Self::has_tool_call(&output) {
                let call = Self::parse_tool_call(&output)?;

                tracing::debug!("Executing tool call: {} with args: {:?}", call.name, call.args);

                // Execute the tool
                let result = self.tools.execute_call(&call).await?;

                tracing::debug!("Tool '{}' returned: {:?}", call.name, result);

                // Inject result into context for next iteration
                self.append_tool_result(&mut input, &call.name, &result);
            } else {
                // No tool call - return the response
                return Ok(output);
            }
        }

        Err(InvokeWithToolsError::MaxIterationsReached(
            self.config.max_iterations,
        ))
    }
}

impl std::fmt::Debug for ToolWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolWrapper")
            .field("config", &self.config)
            .field("tools", &self.tools)
            .finish()
    }
}

/// Errors that can occur during invoke_with_tools
#[derive(Debug, thiserror::Error)]
pub enum InvokeWithToolsError {
    /// Tool error during execution
    #[error("Tool error: {0}")]
    ToolError(#[from] ToolError),

    /// Invocation error from the predictor
    #[error("Invoke error: {0}")]
    InvokeError(String),

    /// Maximum iterations reached
    #[error("Maximum iterations ({0}) reached during tool execution")]
    MaxIterationsReached(usize),
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::json;

    use crate::inference::tools::Tool;

    /// Mock tool that returns a fixed result
    struct MockTool {
        name: String,
        result: Value,
    }

    impl MockTool {
        fn new(name: &str, result: Value) -> Self {
            Self {
                name: name.to_string(),
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
            "Mock tool"
        }

        async fn execute(&self, _args: Value) -> Result<Value, ToolError> {
            Ok(self.result.clone())
        }
    }

    #[test]
    fn test_config_default_values() {
        let config = ToolWrapperConfig::default();
        assert_eq!(config.max_iterations, 5);
        assert_eq!(config.tool_result_key, "context");
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = ToolWrapperConfig::default()
            .with_max_iterations(10)
            .with_tool_result_key("tool_context");

        assert_eq!(config.max_iterations, 10);
        assert_eq!(config.tool_result_key, "tool_context");
    }

    #[test]
    fn test_has_tool_call() {
        assert!(ToolWrapper::has_tool_call(&json!({
            "tool_call": { "name": "test", "args": {} }
        })));

        assert!(!ToolWrapper::has_tool_call(&json!({
            "tool_call": null
        })));

        assert!(!ToolWrapper::has_tool_call(&json!({
            "response": "hello"
        })));
    }

    #[test]
    fn test_parse_tool_call() {
        let output = json!({
            "tool_call": { "name": "get_gold", "args": { "player_id": 1 } }
        });

        let call = ToolWrapper::parse_tool_call(&output).unwrap();
        assert_eq!(call.name, "get_gold");
        assert_eq!(call.args, json!({ "player_id": 1 }));
    }

    #[test]
    fn test_parse_tool_call_missing() {
        let output = json!({ "response": "hello" });
        let result = ToolWrapper::parse_tool_call(&output);
        assert!(result.is_err());
    }

    #[test]
    fn test_append_tool_result_empty_context() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(MockTool::new("test", json!({}))));

        let wrapper = ToolWrapper::new(Arc::new(registry), ToolWrapperConfig::default());

        let mut input = json!({});
        wrapper.append_tool_result(&mut input, "get_gold", &json!({"gold": 500}));

        let context = input["context"].as_str().unwrap();
        assert!(context.contains("get_gold"));
        assert!(context.contains("500"));
    }

    #[test]
    fn test_append_tool_result_existing_context() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(MockTool::new("test", json!({}))));

        let wrapper = ToolWrapper::new(Arc::new(registry), ToolWrapperConfig::default());

        let mut input = json!({
            "context": "Previous context"
        });
        wrapper.append_tool_result(&mut input, "get_gold", &json!({"gold": 500}));

        let context = input["context"].as_str().unwrap();
        assert!(context.starts_with("Previous context"));
        assert!(context.contains("get_gold"));
    }

    #[tokio::test]
    async fn test_invoke_with_no_tool_call_returns_immediately() {
        let registry = Arc::new(ToolRegistry::new());
        let wrapper = ToolWrapper::new(registry, ToolWrapperConfig::default());

        // Mock invoke function that returns output without tool_call
        let invoke_fn = |_input: Value| async {
            Ok::<_, String>(json!({
                "response": "Hello, world!"
            }))
        };

        let result = wrapper
            .invoke_with_fn(invoke_fn, json!({"query": "test"}))
            .await
            .unwrap();

        assert_eq!(result["response"], "Hello, world!");
    }

    #[tokio::test]
    async fn test_invoke_with_tool_call_executes_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(MockTool::new("get_gold", json!({"gold": 500}))));

        let wrapper = ToolWrapper::new(Arc::new(registry), ToolWrapperConfig::default());

        // Track call count
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let invoke_fn = move |_input: Value| {
            let count = call_count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async move {
                if count == 0 {
                    // First call - request tool
                    Ok::<_, String>(json!({
                        "tool_call": { "name": "get_gold", "args": {} }
                    }))
                } else {
                    // Second call - return response
                    Ok(json!({
                        "response": "You have 500 gold"
                    }))
                }
            }
        };

        let result = wrapper
            .invoke_with_fn(invoke_fn, json!({"query": "How much gold?"}))
            .await
            .unwrap();

        assert_eq!(result["response"], "You have 500 gold");
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_invoke_chains_multiple_tool_calls() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(MockTool::new("tool_a", json!({"a": 1}))));
        registry.register(Arc::new(MockTool::new("tool_b", json!({"b": 2}))));

        let wrapper = ToolWrapper::new(Arc::new(registry), ToolWrapperConfig::default());

        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let invoke_fn = move |_input: Value| {
            let count = call_count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async move {
                match count {
                    0 => Ok::<_, String>(json!({
                        "tool_call": { "name": "tool_a", "args": {} }
                    })),
                    1 => Ok(json!({
                        "tool_call": { "name": "tool_b", "args": {} }
                    })),
                    _ => Ok(json!({
                        "response": "Done"
                    })),
                }
            }
        };

        let result = wrapper.invoke_with_fn(invoke_fn, json!({})).await.unwrap();

        assert_eq!(result["response"], "Done");
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_invoke_stops_at_max_iterations() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(MockTool::new("infinite_tool", json!({}))));

        let config = ToolWrapperConfig::default().with_max_iterations(3);
        let wrapper = ToolWrapper::new(Arc::new(registry), config);

        // Always request the same tool (infinite loop)
        let invoke_fn = |_input: Value| async {
            Ok::<_, String>(json!({
                "tool_call": { "name": "infinite_tool", "args": {} }
            }))
        };

        let result = wrapper.invoke_with_fn(invoke_fn, json!({})).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            InvokeWithToolsError::MaxIterationsReached(n) => assert_eq!(n, 3),
            other => panic!("Expected MaxIterationsReached, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_tool_result_appended_to_context_correctly() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(MockTool::new("get_gold", json!({"gold": 500}))));

        let wrapper = ToolWrapper::new(Arc::new(registry), ToolWrapperConfig::default());

        let received_inputs = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let received_inputs_clone = received_inputs.clone();

        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let invoke_fn = move |input: Value| {
            let count = call_count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            received_inputs_clone.lock().unwrap().push(input.clone());
            async move {
                if count == 0 {
                    Ok::<_, String>(json!({
                        "tool_call": { "name": "get_gold", "args": {} }
                    }))
                } else {
                    Ok(json!({ "response": "done" }))
                }
            }
        };

        wrapper.invoke_with_fn(invoke_fn, json!({})).await.unwrap();

        let inputs = received_inputs.lock().unwrap();
        assert_eq!(inputs.len(), 2);

        // First input should have available_tools but no context
        assert!(inputs[0].get("available_tools").is_some());

        // Second input should have context with tool result
        let context = inputs[1]["context"].as_str().unwrap();
        assert!(context.contains("get_gold"));
        assert!(context.contains("500"));
    }

    #[tokio::test]
    async fn test_available_tools_injected() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(MockTool::new("tool1", json!({}))));

        let wrapper = ToolWrapper::new(Arc::new(registry), ToolWrapperConfig::default());

        let received_input = std::sync::Arc::new(std::sync::Mutex::new(None));
        let received_input_clone = received_input.clone();

        let invoke_fn = move |input: Value| {
            *received_input_clone.lock().unwrap() = Some(input.clone());
            async move { Ok::<_, String>(json!({ "response": "done" })) }
        };

        wrapper.invoke_with_fn(invoke_fn, json!({})).await.unwrap();

        let input = received_input.lock().unwrap().clone().unwrap();
        let available_tools = input.get("available_tools").unwrap();
        assert!(available_tools.is_array());

        let tools_arr = available_tools.as_array().unwrap();
        assert_eq!(tools_arr.len(), 1);
        assert_eq!(tools_arr[0]["name"], "tool1");
    }
}
