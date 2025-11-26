//! DSPyEngine registration for Rhai
//!
//! Provides functions to register DSPyEngine methods with a Rhai Engine,
//! enabling Rhai scripts to invoke DSPy modules and manage tools.

use std::sync::Arc;

use rhai::{Array, Dynamic, Engine, FnPtr, Map, AST};

use crate::inference::DSPyEngine;

use super::conversion::{json_to_dynamic, map_to_json};
use super::error::{result_to_dynamic, RhaiIntegrationError};
use super::rhai_tool::RhaiTool;

/// Register DSPyEngine functions with a Rhai Engine
///
/// This function registers the following functions:
///
/// - `dspy_invoke(module_id, input)` - Invoke a module synchronously
/// - `dspy_invoke_with_tools(module_id, input)` - Invoke a module with tool support
/// - `dspy_invoke_with_timeout(module_id, input, timeout_ms)` - Invoke with timeout
/// - `dspy_invoke_with_tools_timeout(module_id, input, timeout_ms)` - Invoke with tools and timeout
/// - `dspy_register_tool(name, description, fn_ptr)` - Register a Rhai function as a tool
/// - `dspy_list_modules()` - List all loaded module IDs
/// - `dspy_list_tools()` - List all registered tool names
/// - `dspy_get_module_info(module_id)` - Get module metadata
///
/// All functions return a result map: `#{ ok: bool, value/error_type/message }`
///
/// # Arguments
///
/// * `rhai_engine` - The Rhai Engine to register functions with
/// * `dspy_engine` - The DSPyEngine instance to wrap
/// * `ast` - The AST for the current script (needed for tool registration)
///
/// # Example
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use rhai::Engine;
/// use ml_crate_dsrs::inference::rhai::register_dspy_engine;
///
/// let mut rhai_engine = Engine::new();
/// let dspy_engine = Arc::new(dspy_engine);
/// let ast = rhai_engine.compile(script)?;
///
/// register_dspy_engine(&mut rhai_engine, dspy_engine.clone(), Arc::new(ast));
///
/// // Now Rhai scripts can call:
/// // let result = dspy_invoke("npc.dialogue", #{ player_message: "Hello" });
/// ```
pub fn register_dspy_engine(rhai_engine: &mut Engine, dspy_engine: Arc<DSPyEngine>, ast: Arc<AST>) {
    // Clone Arc for each closure
    let engine_invoke = dspy_engine.clone();
    let engine_invoke_tools = dspy_engine.clone();
    let engine_invoke_timeout = dspy_engine.clone();
    let engine_invoke_tools_timeout = dspy_engine.clone();
    let engine_register_tool = dspy_engine.clone();
    let engine_list_modules = dspy_engine.clone();
    let engine_list_tools = dspy_engine.clone();
    let engine_get_info = dspy_engine.clone();

    let ast_for_tool = ast.clone();

    // ========================================
    // dspy_invoke(module_id, input) -> Dynamic
    // ========================================
    rhai_engine.register_fn("dspy_invoke", move |module_id: &str, input: Map| -> Dynamic {
        let result = invoke_impl(&engine_invoke, module_id, input);
        result_to_dynamic(result)
    });

    // ========================================
    // dspy_invoke_with_tools(module_id, input) -> Dynamic
    // ========================================
    rhai_engine.register_fn(
        "dspy_invoke_with_tools",
        move |module_id: &str, input: Map| -> Dynamic {
            let result = invoke_with_tools_impl(&engine_invoke_tools, module_id, input);
            result_to_dynamic(result)
        },
    );

    // ========================================
    // dspy_invoke_with_timeout(module_id, input, timeout_ms) -> Dynamic
    // ========================================
    rhai_engine.register_fn(
        "dspy_invoke_with_timeout",
        move |module_id: &str, input: Map, timeout_ms: i64| -> Dynamic {
            let result =
                invoke_with_timeout_impl(&engine_invoke_timeout, module_id, input, timeout_ms as u64);
            result_to_dynamic(result)
        },
    );

    // ========================================
    // dspy_invoke_with_tools_timeout(module_id, input, timeout_ms) -> Dynamic
    // ========================================
    rhai_engine.register_fn(
        "dspy_invoke_with_tools_timeout",
        move |module_id: &str, input: Map, timeout_ms: i64| -> Dynamic {
            let result = invoke_with_tools_timeout_impl(
                &engine_invoke_tools_timeout,
                module_id,
                input,
                timeout_ms as u64,
            );
            result_to_dynamic(result)
        },
    );

    // ========================================
    // dspy_register_tool(name, description, fn_ptr) -> Dynamic
    // ========================================
    let ast_clone = ast_for_tool.clone();
    rhai_engine.register_fn(
        "dspy_register_tool",
        move |name: &str, description: &str, fn_ptr: FnPtr| -> Dynamic {
            let result =
                register_tool_impl(&engine_register_tool, name, description, fn_ptr, ast_clone.clone());
            result_to_dynamic(result)
        },
    );

    // ========================================
    // dspy_list_modules() -> Dynamic
    // ========================================
    rhai_engine.register_fn("dspy_list_modules", move || -> Dynamic {
        let result = list_modules_impl(&engine_list_modules);
        result_to_dynamic(result)
    });

    // ========================================
    // dspy_list_tools() -> Dynamic
    // ========================================
    rhai_engine.register_fn("dspy_list_tools", move || -> Dynamic {
        let result = list_tools_impl(&engine_list_tools);
        result_to_dynamic(result)
    });

    // ========================================
    // dspy_get_module_info(module_id) -> Dynamic
    // ========================================
    rhai_engine.register_fn("dspy_get_module_info", move |module_id: &str| -> Dynamic {
        let result = get_module_info_impl(&engine_get_info, module_id);
        result_to_dynamic(result)
    });
}

/// Implementation of invoke
fn invoke_impl(
    engine: &DSPyEngine,
    module_id: &str,
    input: Map,
) -> Result<Dynamic, RhaiIntegrationError> {
    // Convert Map to JSON Value
    let input_json = map_to_json(input)?;

    // Call the engine
    let result = engine.invoke_sync(module_id, input_json)?;

    // Convert result back to Dynamic
    Ok(json_to_dynamic(result))
}

/// Implementation of invoke_with_tools
fn invoke_with_tools_impl(
    engine: &DSPyEngine,
    module_id: &str,
    input: Map,
) -> Result<Dynamic, RhaiIntegrationError> {
    let input_json = map_to_json(input)?;
    let result = engine.invoke_with_tools_sync(module_id, input_json)?;
    Ok(json_to_dynamic(result))
}

/// Implementation of invoke_with_timeout
fn invoke_with_timeout_impl(
    engine: &DSPyEngine,
    module_id: &str,
    input: Map,
    timeout_ms: u64,
) -> Result<Dynamic, RhaiIntegrationError> {
    let input_json = map_to_json(input)?;
    let result = engine.invoke_sync_with_timeout(module_id, input_json, timeout_ms)?;
    Ok(json_to_dynamic(result))
}

/// Implementation of invoke_with_tools_timeout
fn invoke_with_tools_timeout_impl(
    engine: &DSPyEngine,
    module_id: &str,
    input: Map,
    timeout_ms: u64,
) -> Result<Dynamic, RhaiIntegrationError> {
    let input_json = map_to_json(input)?;
    let result = engine.invoke_with_tools_sync_with_timeout(module_id, input_json, timeout_ms)?;
    Ok(json_to_dynamic(result))
}

/// Implementation of register_tool
fn register_tool_impl(
    engine: &DSPyEngine,
    name: &str,
    description: &str,
    fn_ptr: FnPtr,
    ast: Arc<AST>,
) -> Result<Dynamic, RhaiIntegrationError> {
    // Create RhaiTool
    let tool = RhaiTool::new(name, description, fn_ptr, ast);

    // Register synchronously
    engine.register_tool_sync(Arc::new(tool));

    Ok(Dynamic::from(true))
}

/// Implementation of list_modules
fn list_modules_impl(engine: &DSPyEngine) -> Result<Dynamic, RhaiIntegrationError> {
    let module_ids = engine.module_ids_sync();
    let array: Array = module_ids.into_iter().map(Dynamic::from).collect();
    Ok(Dynamic::from_array(array))
}

/// Implementation of list_tools
fn list_tools_impl(engine: &DSPyEngine) -> Result<Dynamic, RhaiIntegrationError> {
    let tool_names = engine.tool_names_sync();
    let array: Array = tool_names.into_iter().map(Dynamic::from).collect();
    Ok(Dynamic::from_array(array))
}

/// Implementation of get_module_info
fn get_module_info_impl(
    engine: &DSPyEngine,
    module_id: &str,
) -> Result<Dynamic, RhaiIntegrationError> {
    let module = engine
        .get_module_sync(module_id)
        .ok_or_else(|| RhaiIntegrationError::module_not_found(module_id))?;

    // Build info map
    let mut map = Map::new();
    map.insert("module_id".into(), Dynamic::from(module.module_id.clone()));
    map.insert(
        "signature_name".into(),
        Dynamic::from(module.signature_name.clone()),
    );
    map.insert(
        "instruction".into(),
        Dynamic::from(module.instruction.clone()),
    );
    map.insert(
        "predictor_type".into(),
        Dynamic::from(format!("{:?}", module.predictor_type)),
    );
    map.insert(
        "tool_enabled".into(),
        Dynamic::from(module.tool_enabled),
    );
    map.insert("demo_count".into(), Dynamic::from(module.demos.len() as i64));

    // Add metadata
    let metadata = &module.metadata;
    let mut meta_map = Map::new();
    meta_map.insert("version".into(), Dynamic::from(metadata.version.clone()));
    meta_map.insert(
        "optimizer".into(),
        Dynamic::from(metadata.optimizer.clone()),
    );
    meta_map.insert(
        "optimized_at".into(),
        metadata
            .optimized_at
            .as_ref()
            .map(|s| Dynamic::from(s.clone()))
            .unwrap_or(Dynamic::UNIT),
    );
    meta_map.insert(
        "metric_score".into(),
        metadata
            .metric_score
            .map(|s| Dynamic::from(s as f64))
            .unwrap_or(Dynamic::UNIT),
    );
    map.insert("metadata".into(), Dynamic::from_map(meta_map));

    Ok(Dynamic::from_map(map))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full integration tests require a working DSPyEngine with CandleAdapter,
    // which is covered in tests/rhai_integration_tests.rs.
    // Here we test the conversion utilities.

    #[test]
    fn test_map_to_json_for_invoke() {
        let mut map = Map::new();
        map.insert("question".into(), Dynamic::from("What is Rust?"));
        map.insert("context".into(), Dynamic::from("Programming"));

        let json = map_to_json(map).unwrap();

        assert_eq!(json["question"], "What is Rust?");
        assert_eq!(json["context"], "Programming");
    }

    #[test]
    fn test_result_to_dynamic_format() {
        // Success case
        let success: Result<i64, RhaiIntegrationError> = Ok(42);
        let dynamic = result_to_dynamic(success);
        let map: Map = dynamic.cast();

        assert_eq!(map.get("ok").unwrap().clone().cast::<bool>(), true);
        assert_eq!(map.get("value").unwrap().clone().cast::<i64>(), 42);

        // Error case
        let error: Result<i64, RhaiIntegrationError> =
            Err(RhaiIntegrationError::module_not_found("test"));
        let dynamic = result_to_dynamic(error);
        let map: Map = dynamic.cast();

        assert_eq!(map.get("ok").unwrap().clone().cast::<bool>(), false);
        assert_eq!(
            map.get("error_type").unwrap().clone().cast::<String>(),
            "ModuleNotFound"
        );
    }
}
