//! Rhai scripting integration for DSPyEngine
//!
//! This module provides the integration layer between Rhai scripts and the DSPyEngine,
//! enabling game servers to invoke LLM modules and register tools from Rhai scripts.
//!
//! # Features
//!
//! - **JSON ↔ Dynamic conversion**: Bidirectional conversion between `serde_json::Value` and Rhai `Dynamic`
//! - **RhaiTool**: Wrap Rhai functions as tools callable by DSPy modules
//! - **Engine registration**: Register DSPyEngine functions with a Rhai Engine
//! - **Structured errors**: Rhai-friendly error types for pattern matching in scripts
//!
//! # Example
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use rhai::Engine;
//! use ml_crate_dsrs::inference::rhai::register_dspy_engine;
//!
//! // Create Rhai engine and compile script
//! let mut rhai_engine = Engine::new();
//! let ast = rhai_engine.compile(r#"
//!     // Define a tool
//!     fn get_player_gold(args) {
//!         #{ gold: 500 }
//!     }
//!
//!     // Register the tool
//!     dspy_register_tool("get_player_gold", "Get player's gold", Fn("get_player_gold"));
//!
//!     // Invoke a module
//!     let result = dspy_invoke("npc.dialogue", #{
//!         player_message: "Hello there!"
//!     });
//!
//!     if result.ok {
//!         print(result.value.response);
//!     } else {
//!         print("Error: " + result.message);
//!     }
//! "#)?;
//!
//! // Register DSPyEngine with Rhai
//! register_dspy_engine(&mut rhai_engine, dspy_engine.clone(), Arc::new(ast.clone()));
//!
//! // Run the script
//! rhai_engine.run_ast(&ast)?;
//! ```
//!
//! # Available Rhai Functions
//!
//! After calling `register_dspy_engine`, these functions are available in Rhai:
//!
//! | Function | Description |
//! |----------|-------------|
//! | `dspy_invoke(module_id, input)` | Invoke a module synchronously |
//! | `dspy_invoke_with_tools(module_id, input)` | Invoke with tool support |
//! | `dspy_invoke_with_timeout(module_id, input, timeout_ms)` | Invoke with timeout |
//! | `dspy_invoke_with_tools_timeout(module_id, input, timeout_ms)` | Invoke with tools and timeout |
//! | `dspy_register_tool(name, description, fn_ptr)` | Register a Rhai function as a tool |
//! | `dspy_list_modules()` | List all loaded module IDs |
//! | `dspy_list_tools()` | List all registered tool names |
//! | `dspy_get_module_info(module_id)` | Get module metadata |
//!
//! All functions return a result map: `#{ ok: bool, value/error_type/message }`

pub mod conversion;
pub mod error;
pub mod registration;
pub mod rhai_tool;

// Re-export commonly used types and functions
pub use conversion::{dynamic_to_json, json_to_dynamic, map_to_json};
pub use error::{result_to_dynamic, RhaiConversionError, RhaiIntegrationError};
pub use registration::register_dspy_engine;
pub use rhai_tool::RhaiTool;

// Re-export Rhai types that users will commonly need
pub use rhai::{Array, Dynamic, Engine, FnPtr, Map, AST};
