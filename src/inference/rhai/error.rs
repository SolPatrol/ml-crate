//! Rhai integration error types
//!
//! Defines error types for Rhai ↔ DSPyEngine integration.

use rhai::Dynamic;
use thiserror::Error;

/// Errors that can occur during JSON ↔ Rhai Dynamic conversion
#[derive(Debug, Error)]
pub enum RhaiConversionError {
    /// Dynamic type cannot be converted to JSON
    #[error("Unsupported type for JSON conversion: {0}")]
    UnsupportedType(String),

    /// Map has non-string keys
    #[error("Map contains non-string keys, cannot convert to JSON object")]
    NonStringKey,
}

impl RhaiConversionError {
    /// Create an unsupported type error
    pub fn unsupported_type(type_name: impl Into<String>) -> Self {
        Self::UnsupportedType(type_name.into())
    }
}

/// Structured errors for Rhai scripts
///
/// These errors can be returned to Rhai in a structured format
/// that scripts can pattern match on.
#[derive(Debug, Error)]
pub enum RhaiIntegrationError {
    /// Module not found in the registry
    #[error("Module not found: {module_id}")]
    ModuleNotFound { module_id: String },

    /// Signature not found in the registry
    #[error("Signature not found: {signature_name}")]
    SignatureNotFound { signature_name: String },

    /// Tools not enabled for this module
    #[error("Tools not enabled for module: {module_id}")]
    ToolsNotEnabled { module_id: String },

    /// Tool not found in the registry
    #[error("Tool not found: {tool_name}")]
    ToolNotFound { tool_name: String },

    /// Tool execution failed
    #[error("Tool execution failed for '{tool_name}': {message}")]
    ToolExecutionFailed { tool_name: String, message: String },

    /// Operation timed out
    #[error("Operation timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    /// Inference error during model execution
    #[error("Inference error: {message}")]
    InferenceError { message: String },

    /// Conversion error
    #[error("Conversion error: {message}")]
    ConversionError { message: String },

    /// Runtime error
    #[error("Runtime error: {message}")]
    RuntimeError { message: String },
}

impl RhaiIntegrationError {
    /// Get the error type name for Rhai
    pub fn error_type(&self) -> &'static str {
        match self {
            Self::ModuleNotFound { .. } => "ModuleNotFound",
            Self::SignatureNotFound { .. } => "SignatureNotFound",
            Self::ToolsNotEnabled { .. } => "ToolsNotEnabled",
            Self::ToolNotFound { .. } => "ToolNotFound",
            Self::ToolExecutionFailed { .. } => "ToolExecutionFailed",
            Self::Timeout { .. } => "Timeout",
            Self::InferenceError { .. } => "InferenceError",
            Self::ConversionError { .. } => "ConversionError",
            Self::RuntimeError { .. } => "RuntimeError",
        }
    }

    /// Create a module not found error
    pub fn module_not_found(module_id: impl Into<String>) -> Self {
        Self::ModuleNotFound {
            module_id: module_id.into(),
        }
    }

    /// Create a signature not found error
    pub fn signature_not_found(signature_name: impl Into<String>) -> Self {
        Self::SignatureNotFound {
            signature_name: signature_name.into(),
        }
    }

    /// Create a tools not enabled error
    pub fn tools_not_enabled(module_id: impl Into<String>) -> Self {
        Self::ToolsNotEnabled {
            module_id: module_id.into(),
        }
    }

    /// Create a tool not found error
    pub fn tool_not_found(tool_name: impl Into<String>) -> Self {
        Self::ToolNotFound {
            tool_name: tool_name.into(),
        }
    }

    /// Create a tool execution failed error
    pub fn tool_execution_failed(tool_name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ToolExecutionFailed {
            tool_name: tool_name.into(),
            message: message.into(),
        }
    }

    /// Create a timeout error
    pub fn timeout(timeout_ms: u64) -> Self {
        Self::Timeout { timeout_ms }
    }

    /// Create an inference error
    pub fn inference(message: impl Into<String>) -> Self {
        Self::InferenceError {
            message: message.into(),
        }
    }

    /// Create a conversion error
    pub fn conversion(message: impl Into<String>) -> Self {
        Self::ConversionError {
            message: message.into(),
        }
    }

    /// Create a runtime error
    pub fn runtime(message: impl Into<String>) -> Self {
        Self::RuntimeError {
            message: message.into(),
        }
    }
}

/// Convert a Result to a Rhai-friendly Dynamic map
///
/// Returns a map in the format:
/// - Success: `#{ ok: true, value: ... }`
/// - Error: `#{ ok: false, error_type: "...", message: "..." }`
pub fn result_to_dynamic<T, E>(result: Result<T, E>) -> Dynamic
where
    T: Into<Dynamic>,
    E: Into<RhaiIntegrationError>,
{
    use rhai::Map;

    let mut map = Map::new();

    match result {
        Ok(value) => {
            map.insert("ok".into(), Dynamic::from(true));
            map.insert("value".into(), value.into());
        }
        Err(err) => {
            let err = err.into();
            map.insert("ok".into(), Dynamic::from(false));
            map.insert("error_type".into(), Dynamic::from(err.error_type()));
            map.insert("message".into(), Dynamic::from(err.to_string()));
        }
    }

    Dynamic::from_map(map)
}

/// Convert a DSPyEngineError to a RhaiIntegrationError
impl From<crate::inference::error::DSPyEngineError> for RhaiIntegrationError {
    fn from(err: crate::inference::error::DSPyEngineError) -> Self {
        use crate::inference::error::DSPyEngineError;

        match err {
            DSPyEngineError::ModuleNotFound(module_id) => {
                RhaiIntegrationError::ModuleNotFound { module_id }
            }
            DSPyEngineError::SignatureNotFound(signature_name) => {
                RhaiIntegrationError::SignatureNotFound { signature_name }
            }
            DSPyEngineError::ToolsNotEnabled(module_id) => {
                RhaiIntegrationError::ToolsNotEnabled { module_id }
            }
            DSPyEngineError::Timeout(timeout_ms) => RhaiIntegrationError::Timeout { timeout_ms },
            DSPyEngineError::InferenceError(msg) => {
                RhaiIntegrationError::InferenceError { message: msg }
            }
            DSPyEngineError::ToolError(tool_err) => match tool_err {
                crate::inference::tools::ToolError::NotFound(name) => {
                    RhaiIntegrationError::ToolNotFound { tool_name: name }
                }
                crate::inference::tools::ToolError::ExecutionFailed(msg) => {
                    RhaiIntegrationError::ToolExecutionFailed {
                        tool_name: "unknown".to_string(),
                        message: msg,
                    }
                }
                other => RhaiIntegrationError::RuntimeError {
                    message: other.to_string(),
                },
            },
            other => RhaiIntegrationError::RuntimeError {
                message: other.to_string(),
            },
        }
    }
}

/// Convert RhaiConversionError to RhaiIntegrationError
impl From<RhaiConversionError> for RhaiIntegrationError {
    fn from(err: RhaiConversionError) -> Self {
        RhaiIntegrationError::ConversionError {
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_error_display() {
        let err = RhaiConversionError::unsupported_type("FnPtr");
        assert!(err.to_string().contains("FnPtr"));

        let err = RhaiConversionError::NonStringKey;
        assert!(err.to_string().contains("non-string keys"));
    }

    #[test]
    fn test_integration_error_types() {
        assert_eq!(
            RhaiIntegrationError::module_not_found("test").error_type(),
            "ModuleNotFound"
        );
        assert_eq!(
            RhaiIntegrationError::signature_not_found("test").error_type(),
            "SignatureNotFound"
        );
        assert_eq!(
            RhaiIntegrationError::tools_not_enabled("test").error_type(),
            "ToolsNotEnabled"
        );
        assert_eq!(
            RhaiIntegrationError::tool_not_found("test").error_type(),
            "ToolNotFound"
        );
        assert_eq!(
            RhaiIntegrationError::tool_execution_failed("test", "failed").error_type(),
            "ToolExecutionFailed"
        );
        assert_eq!(
            RhaiIntegrationError::timeout(1000).error_type(),
            "Timeout"
        );
        assert_eq!(
            RhaiIntegrationError::inference("test").error_type(),
            "InferenceError"
        );
    }

    #[test]
    fn test_result_to_dynamic_success() {
        let result: Result<i64, RhaiIntegrationError> = Ok(42);
        let dynamic = result_to_dynamic(result);

        let map = dynamic.cast::<rhai::Map>();
        assert_eq!(map.get("ok").unwrap().clone().cast::<bool>(), true);
        assert_eq!(map.get("value").unwrap().clone().cast::<i64>(), 42);
    }

    #[test]
    fn test_result_to_dynamic_error() {
        let result: Result<i64, RhaiIntegrationError> =
            Err(RhaiIntegrationError::module_not_found("test_module"));
        let dynamic = result_to_dynamic(result);

        let map = dynamic.cast::<rhai::Map>();
        assert_eq!(map.get("ok").unwrap().clone().cast::<bool>(), false);
        assert_eq!(
            map.get("error_type").unwrap().clone().cast::<String>(),
            "ModuleNotFound"
        );
        assert!(map
            .get("message")
            .unwrap()
            .clone()
            .cast::<String>()
            .contains("test_module"));
    }
}
