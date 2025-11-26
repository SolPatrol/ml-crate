//! JSON ↔ Rhai Dynamic conversion helpers
//!
//! Provides bidirectional conversion between serde_json::Value and Rhai Dynamic types.
//! These conversions are essential for passing data between Rhai scripts and the DSPyEngine.

use rhai::{Array, Dynamic, ImmutableString, Map};
use serde_json::{Number, Value};

use super::error::RhaiConversionError;

/// Convert a serde_json::Value to a Rhai Dynamic
///
/// # Type Mappings
///
/// - `null` → `()` (unit type)
/// - `bool` → `bool`
/// - `number` (integer) → `i64`
/// - `number` (float) → `f64`
/// - `string` → `ImmutableString`
/// - `array` → `Array` (recursive)
/// - `object` → `Map` (recursive)
///
/// # Example
///
/// ```rust,ignore
/// use serde_json::json;
/// use ml_crate_dsrs::inference::rhai::json_to_dynamic;
///
/// let json = json!({
///     "name": "test",
///     "count": 42,
///     "items": [1, 2, 3]
/// });
///
/// let dynamic = json_to_dynamic(json);
/// ```
pub fn json_to_dynamic(value: Value) -> Dynamic {
    match value {
        Value::Null => Dynamic::UNIT,
        Value::Bool(b) => Dynamic::from(b),
        Value::Number(n) => {
            // Prefer i64 for integers, fallback to f64 for floats
            if let Some(i) = n.as_i64() {
                Dynamic::from(i)
            } else if let Some(f) = n.as_f64() {
                Dynamic::from(f)
            } else {
                // Should not happen, but handle gracefully
                Dynamic::UNIT
            }
        }
        Value::String(s) => Dynamic::from(s),
        Value::Array(arr) => {
            let rhai_arr: Array = arr.into_iter().map(json_to_dynamic).collect();
            Dynamic::from_array(rhai_arr)
        }
        Value::Object(obj) => {
            let mut map = Map::new();
            for (k, v) in obj {
                map.insert(k.into(), json_to_dynamic(v));
            }
            Dynamic::from_map(map)
        }
    }
}

/// Convert a Rhai Dynamic to a serde_json::Value
///
/// # Type Mappings
///
/// - `()` (unit) → `null`
/// - `bool` → `bool`
/// - `i64` (INT) → `number`
/// - `f64` (FLOAT) → `number`
/// - `ImmutableString`/`String` → `string`
/// - `Array` → `array` (recursive)
/// - `Map` → `object` (recursive, validates string keys)
///
/// # Errors
///
/// Returns `RhaiConversionError::UnsupportedType` for types that cannot be
/// represented in JSON (e.g., functions, custom types).
///
/// Returns `RhaiConversionError::NonStringKey` if a map contains non-string keys.
///
/// # Example
///
/// ```rust,ignore
/// use rhai::Dynamic;
/// use ml_crate_dsrs::inference::rhai::dynamic_to_json;
///
/// let dynamic = Dynamic::from(42_i64);
/// let json = dynamic_to_json(dynamic)?;
/// assert_eq!(json, serde_json::json!(42));
/// ```
pub fn dynamic_to_json(dynamic: Dynamic) -> Result<Value, RhaiConversionError> {
    // Handle unit type
    if dynamic.is_unit() {
        return Ok(Value::Null);
    }

    // Handle bool
    if dynamic.is_bool() {
        return Ok(Value::Bool(dynamic.as_bool().unwrap()));
    }

    // Handle integer (INT = i64)
    if dynamic.is_int() {
        let i = dynamic.as_int().unwrap();
        return Ok(Value::Number(Number::from(i)));
    }

    // Handle float
    if dynamic.is_float() {
        let f = dynamic.as_float().unwrap();
        return Number::from_f64(f)
            .map(Value::Number)
            .ok_or_else(|| RhaiConversionError::unsupported_type("non-finite float"));
    }

    // Handle string (both String and ImmutableString)
    if dynamic.is_string() {
        let s: ImmutableString = dynamic.cast();
        return Ok(Value::String(s.to_string()));
    }

    // Handle array
    if dynamic.is_array() {
        let arr: Array = dynamic.cast();
        let json_arr: Result<Vec<Value>, RhaiConversionError> =
            arr.into_iter().map(dynamic_to_json).collect();
        return Ok(Value::Array(json_arr?));
    }

    // Handle map
    if dynamic.is_map() {
        let map: Map = dynamic.cast();
        return map_to_json(map);
    }

    // Handle char as single-character string
    if dynamic.is_char() {
        let c: char = dynamic.cast();
        return Ok(Value::String(c.to_string()));
    }

    // Unsupported type
    Err(RhaiConversionError::unsupported_type(dynamic.type_name()))
}

/// Convert a Rhai Map to a JSON object
///
/// Validates that all keys are strings.
///
/// # Errors
///
/// Returns `RhaiConversionError::NonStringKey` if any key is not a string.
pub fn map_to_json(map: Map) -> Result<Value, RhaiConversionError> {
    let mut obj = serde_json::Map::new();

    for (key, value) in map {
        // Keys in Rhai Map are ImmutableString (SmartString)
        let key_str = key.to_string();
        let json_value = dynamic_to_json(value)?;
        obj.insert(key_str, json_value);
    }

    Ok(Value::Object(obj))
}

/// Convert a JSON Value reference to a Rhai Dynamic (non-consuming)
///
/// Same as `json_to_dynamic` but takes a reference and clones the value.
pub fn json_ref_to_dynamic(value: &Value) -> Dynamic {
    json_to_dynamic(value.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ============================================
    // json_to_dynamic tests
    // ============================================

    #[test]
    fn test_json_to_dynamic_null() {
        let json = Value::Null;
        let dynamic = json_to_dynamic(json);
        assert!(dynamic.is_unit());
    }

    #[test]
    fn test_json_to_dynamic_bool() {
        let json_true = json!(true);
        let json_false = json!(false);

        let dyn_true = json_to_dynamic(json_true);
        let dyn_false = json_to_dynamic(json_false);

        assert!(dyn_true.is_bool());
        assert!(dyn_false.is_bool());
        assert_eq!(dyn_true.as_bool().unwrap(), true);
        assert_eq!(dyn_false.as_bool().unwrap(), false);
    }

    #[test]
    fn test_json_to_dynamic_integer() {
        let json = json!(42);
        let dynamic = json_to_dynamic(json);

        assert!(dynamic.is_int());
        assert_eq!(dynamic.as_int().unwrap(), 42);
    }

    #[test]
    fn test_json_to_dynamic_negative_integer() {
        let json = json!(-100);
        let dynamic = json_to_dynamic(json);

        assert!(dynamic.is_int());
        assert_eq!(dynamic.as_int().unwrap(), -100);
    }

    #[test]
    fn test_json_to_dynamic_float() {
        let json = json!(3.14159);
        let dynamic = json_to_dynamic(json);

        assert!(dynamic.is_float());
        assert!((dynamic.as_float().unwrap() - 3.14159).abs() < f64::EPSILON);
    }

    #[test]
    fn test_json_to_dynamic_string() {
        let json = json!("hello world");
        let dynamic = json_to_dynamic(json);

        assert!(dynamic.is_string());
        let s: ImmutableString = dynamic.cast();
        assert_eq!(s.as_str(), "hello world");
    }

    #[test]
    fn test_json_to_dynamic_empty_string() {
        let json = json!("");
        let dynamic = json_to_dynamic(json);

        assert!(dynamic.is_string());
        let s: ImmutableString = dynamic.cast();
        assert_eq!(s.as_str(), "");
    }

    #[test]
    fn test_json_to_dynamic_array() {
        let json = json!([1, 2, 3]);
        let dynamic = json_to_dynamic(json);

        assert!(dynamic.is_array());
        let arr: Array = dynamic.cast();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0].as_int().unwrap(), 1);
        assert_eq!(arr[1].as_int().unwrap(), 2);
        assert_eq!(arr[2].as_int().unwrap(), 3);
    }

    #[test]
    fn test_json_to_dynamic_empty_array() {
        let json = json!([]);
        let dynamic = json_to_dynamic(json);

        assert!(dynamic.is_array());
        let arr: Array = dynamic.cast();
        assert!(arr.is_empty());
    }

    #[test]
    fn test_json_to_dynamic_mixed_array() {
        let json = json!([1, "two", true, null]);
        let dynamic = json_to_dynamic(json);

        assert!(dynamic.is_array());
        let arr: Array = dynamic.cast();
        assert_eq!(arr.len(), 4);
        assert!(arr[0].is_int());
        assert!(arr[1].is_string());
        assert!(arr[2].is_bool());
        assert!(arr[3].is_unit());
    }

    #[test]
    fn test_json_to_dynamic_object() {
        let json = json!({
            "name": "test",
            "count": 42
        });
        let dynamic = json_to_dynamic(json);

        assert!(dynamic.is_map());
        let map: Map = dynamic.cast();
        assert_eq!(map.len(), 2);

        let name = map.get("name").unwrap();
        assert!(name.is_string());

        let count = map.get("count").unwrap();
        assert!(count.is_int());
        assert_eq!(count.as_int().unwrap(), 42);
    }

    #[test]
    fn test_json_to_dynamic_empty_object() {
        let json = json!({});
        let dynamic = json_to_dynamic(json);

        assert!(dynamic.is_map());
        let map: Map = dynamic.cast();
        assert!(map.is_empty());
    }

    #[test]
    fn test_json_to_dynamic_nested_object() {
        let json = json!({
            "outer": {
                "inner": {
                    "value": 123
                }
            }
        });
        let dynamic = json_to_dynamic(json);

        assert!(dynamic.is_map());
        let outer_map: Map = dynamic.cast();

        let outer = outer_map.get("outer").unwrap();
        assert!(outer.is_map());

        let inner_map: Map = outer.clone().cast();
        let inner = inner_map.get("inner").unwrap();
        assert!(inner.is_map());

        let value_map: Map = inner.clone().cast();
        let value = value_map.get("value").unwrap();
        assert_eq!(value.as_int().unwrap(), 123);
    }

    #[test]
    fn test_json_to_dynamic_array_of_objects() {
        let json = json!([
            {"id": 1, "name": "first"},
            {"id": 2, "name": "second"}
        ]);
        let dynamic = json_to_dynamic(json);

        assert!(dynamic.is_array());
        let arr: Array = dynamic.cast();
        assert_eq!(arr.len(), 2);

        for item in arr {
            assert!(item.is_map());
        }
    }

    // ============================================
    // dynamic_to_json tests
    // ============================================

    #[test]
    fn test_dynamic_to_json_unit() {
        let dynamic = Dynamic::UNIT;
        let json = dynamic_to_json(dynamic).unwrap();
        assert!(json.is_null());
    }

    #[test]
    fn test_dynamic_to_json_bool() {
        let dyn_true = Dynamic::from(true);
        let dyn_false = Dynamic::from(false);

        assert_eq!(dynamic_to_json(dyn_true).unwrap(), json!(true));
        assert_eq!(dynamic_to_json(dyn_false).unwrap(), json!(false));
    }

    #[test]
    fn test_dynamic_to_json_integer() {
        let dynamic = Dynamic::from(42_i64);
        let json = dynamic_to_json(dynamic).unwrap();

        assert!(json.is_number());
        assert_eq!(json.as_i64().unwrap(), 42);
    }

    #[test]
    fn test_dynamic_to_json_float() {
        let dynamic = Dynamic::from(3.14159_f64);
        let json = dynamic_to_json(dynamic).unwrap();

        assert!(json.is_number());
        assert!((json.as_f64().unwrap() - 3.14159).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dynamic_to_json_string() {
        let dynamic = Dynamic::from("hello world");
        let json = dynamic_to_json(dynamic).unwrap();

        assert!(json.is_string());
        assert_eq!(json.as_str().unwrap(), "hello world");
    }

    #[test]
    fn test_dynamic_to_json_char() {
        let dynamic = Dynamic::from('x');
        let json = dynamic_to_json(dynamic).unwrap();

        assert!(json.is_string());
        assert_eq!(json.as_str().unwrap(), "x");
    }

    #[test]
    fn test_dynamic_to_json_array() {
        let arr: Array = vec![Dynamic::from(1_i64), Dynamic::from(2_i64), Dynamic::from(3_i64)];
        let dynamic = Dynamic::from_array(arr);
        let json = dynamic_to_json(dynamic).unwrap();

        assert_eq!(json, json!([1, 2, 3]));
    }

    #[test]
    fn test_dynamic_to_json_map() {
        let mut map = Map::new();
        map.insert("name".into(), Dynamic::from("test"));
        map.insert("count".into(), Dynamic::from(42_i64));

        let dynamic = Dynamic::from_map(map);
        let json = dynamic_to_json(dynamic).unwrap();

        assert!(json.is_object());
        assert_eq!(json["name"], "test");
        assert_eq!(json["count"], 42);
    }

    #[test]
    fn test_dynamic_to_json_nested() {
        let mut inner = Map::new();
        inner.insert("value".into(), Dynamic::from(123_i64));

        let mut outer = Map::new();
        outer.insert("inner".into(), Dynamic::from_map(inner));

        let dynamic = Dynamic::from_map(outer);
        let json = dynamic_to_json(dynamic).unwrap();

        assert_eq!(json["inner"]["value"], 123);
    }

    // ============================================
    // Roundtrip tests
    // ============================================

    #[test]
    fn test_roundtrip_simple_values() {
        let values = vec![
            json!(null),
            json!(true),
            json!(false),
            json!(42),
            json!(-100),
            json!(3.14),
            json!("hello"),
            json!(""),
        ];

        for original in values {
            let dynamic = json_to_dynamic(original.clone());
            let roundtrip = dynamic_to_json(dynamic).unwrap();
            assert_eq!(original, roundtrip, "Roundtrip failed for {:?}", original);
        }
    }

    #[test]
    fn test_roundtrip_array() {
        let original = json!([1, 2, 3, "four", true, null]);
        let dynamic = json_to_dynamic(original.clone());
        let roundtrip = dynamic_to_json(dynamic).unwrap();
        assert_eq!(original, roundtrip);
    }

    #[test]
    fn test_roundtrip_object() {
        let original = json!({
            "string": "value",
            "number": 42,
            "float": 3.14,
            "bool": true,
            "null": null
        });
        let dynamic = json_to_dynamic(original.clone());
        let roundtrip = dynamic_to_json(dynamic).unwrap();
        assert_eq!(original, roundtrip);
    }

    #[test]
    fn test_roundtrip_nested_complex() {
        let original = json!({
            "users": [
                {"id": 1, "name": "Alice", "active": true},
                {"id": 2, "name": "Bob", "active": false}
            ],
            "metadata": {
                "version": "1.0",
                "nested": {
                    "deep": [1, 2, 3]
                }
            },
            "tags": ["a", "b", "c"]
        });
        let dynamic = json_to_dynamic(original.clone());
        let roundtrip = dynamic_to_json(dynamic).unwrap();
        assert_eq!(original, roundtrip);
    }

    // ============================================
    // map_to_json tests
    // ============================================

    #[test]
    fn test_map_to_json_empty() {
        let map = Map::new();
        let json = map_to_json(map).unwrap();
        assert_eq!(json, json!({}));
    }

    #[test]
    fn test_map_to_json_simple() {
        let mut map = Map::new();
        map.insert("key".into(), Dynamic::from("value"));
        let json = map_to_json(map).unwrap();
        assert_eq!(json, json!({"key": "value"}));
    }
}
