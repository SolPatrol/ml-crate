//! Module manifest types for indexing and loading optimized modules
//!
//! The manifest provides a central registry for fast module lookup and
//! supports hot reload by tracking file hashes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use super::error::{DSPyEngineError, Result};
use super::module::OptimizedModule;

/// Entry in the module manifest
///
/// Describes where to find a module and provides metadata for
/// caching and categorization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleEntry {
    /// Relative path from modules/ directory to the module JSON file
    pub path: String,

    /// File hash for hot reload detection (SHA-256)
    #[serde(default)]
    pub hash: Option<String>,

    /// Tags for categorization and filtering
    #[serde(default)]
    pub tags: Vec<String>,
}

impl ModuleEntry {
    /// Create a new module entry
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            hash: None,
            tags: Vec::new(),
        }
    }

    /// Create entry with hash
    pub fn with_hash(path: impl Into<String>, hash: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            hash: Some(hash.into()),
            tags: Vec::new(),
        }
    }

    /// Add tags to the entry
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Check if entry has a specific tag
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

/// Central manifest for module registry
///
/// The manifest indexes all available modules and is typically
/// loaded from a manifest.json file in the modules directory.
///
/// # Example manifest.json
///
/// ```json
/// {
///   "version": "1.0",
///   "modules": {
///     "npc.dialogue.casual": {
///       "path": "npc/dialogue_casual.json",
///       "hash": "sha256:abc123...",
///       "tags": ["npc", "dialogue"]
///     }
///   }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleManifest {
    /// Manifest format version
    pub version: String,

    /// Map of module_id → ModuleEntry
    #[serde(default)]
    pub modules: HashMap<String, ModuleEntry>,
}

impl Default for ModuleManifest {
    fn default() -> Self {
        Self {
            version: "1.0".to_string(),
            modules: HashMap::new(),
        }
    }
}

impl ModuleManifest {
    /// Create a new empty manifest
    pub fn new() -> Self {
        Self::default()
    }

    /// Create manifest with specific version
    pub fn with_version(version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            modules: HashMap::new(),
        }
    }

    /// Add a module entry
    pub fn add_module(&mut self, module_id: impl Into<String>, entry: ModuleEntry) {
        self.modules.insert(module_id.into(), entry);
    }

    /// Get a module entry by ID
    pub fn get(&self, module_id: &str) -> Option<&ModuleEntry> {
        self.modules.get(module_id)
    }

    /// Check if a module exists
    pub fn contains(&self, module_id: &str) -> bool {
        self.modules.contains_key(module_id)
    }

    /// Get all module IDs
    pub fn module_ids(&self) -> Vec<&str> {
        self.modules.keys().map(|s| s.as_str()).collect()
    }

    /// Get modules with a specific tag
    pub fn modules_with_tag(&self, tag: &str) -> Vec<&str> {
        self.modules
            .iter()
            .filter(|(_, entry)| entry.has_tag(tag))
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Get the number of modules
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Check if the manifest is empty
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Remove a module entry
    pub fn remove(&mut self, module_id: &str) -> Option<ModuleEntry> {
        self.modules.remove(module_id)
    }
}

/// Load JSON from a file path
///
/// Generic helper for loading any deserializable type from JSON.
pub fn load_json<T>(path: &Path) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let content = std::fs::read_to_string(path)?;
    let value = serde_json::from_str(&content)?;
    Ok(value)
}

/// Load JSON from a file path with better error messages
pub fn load_json_with_context<T>(path: &Path, context: &str) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let content = std::fs::read_to_string(path).map_err(|e| {
        DSPyEngineError::IoError(std::io::Error::new(
            e.kind(),
            format!("{}: {}", context, e),
        ))
    })?;

    serde_json::from_str(&content).map_err(|e| {
        DSPyEngineError::ParseError(format!("{}: {} at line {}", context, e, e.line()))
    })
}

/// Save JSON to a file path
pub fn save_json<T>(path: &Path, value: &T) -> Result<()>
where
    T: Serialize,
{
    let content = serde_json::to_string_pretty(value)?;
    std::fs::write(path, content)?;
    Ok(())
}

/// Module loader for loading modules from a manifest
pub struct ModuleLoader {
    /// Base directory for modules
    modules_dir: std::path::PathBuf,
    /// The manifest
    manifest: ModuleManifest,
}

impl ModuleLoader {
    /// Create a new module loader
    pub fn new(modules_dir: impl Into<std::path::PathBuf>) -> Result<Self> {
        let modules_dir = modules_dir.into();
        let manifest_path = modules_dir.join("manifest.json");

        let manifest = if manifest_path.exists() {
            load_json_with_context(&manifest_path, "Loading manifest")?
        } else {
            ModuleManifest::default()
        };

        Ok(Self {
            modules_dir,
            manifest,
        })
    }

    /// Create loader with a pre-loaded manifest
    pub fn with_manifest(
        modules_dir: impl Into<std::path::PathBuf>,
        manifest: ModuleManifest,
    ) -> Self {
        Self {
            modules_dir: modules_dir.into(),
            manifest,
        }
    }

    /// Get the manifest
    pub fn manifest(&self) -> &ModuleManifest {
        &self.manifest
    }

    /// Load a single module by ID
    pub fn load_module(&self, module_id: &str) -> Result<OptimizedModule> {
        let entry = self
            .manifest
            .get(module_id)
            .ok_or_else(|| DSPyEngineError::module_not_found(module_id))?;

        let module_path = self.modules_dir.join(&entry.path);
        load_json_with_context(&module_path, &format!("Loading module '{}'", module_id))
    }

    /// Load all modules
    pub fn load_all(&self) -> Result<HashMap<String, OptimizedModule>> {
        let mut modules = HashMap::new();

        for (module_id, entry) in &self.manifest.modules {
            let module_path = self.modules_dir.join(&entry.path);
            let module: OptimizedModule = load_json_with_context(
                &module_path,
                &format!("Loading module '{}'", module_id),
            )?;
            modules.insert(module_id.clone(), module);
        }

        Ok(modules)
    }

    /// Reload the manifest from disk
    pub fn reload_manifest(&mut self) -> Result<()> {
        let manifest_path = self.modules_dir.join("manifest.json");
        self.manifest = load_json_with_context(&manifest_path, "Reloading manifest")?;
        Ok(())
    }

    /// Check if a module file has changed (by hash)
    pub fn has_module_changed(&self, module_id: &str, current_hash: &str) -> bool {
        self.manifest
            .get(module_id)
            .and_then(|e| e.hash.as_ref())
            .map(|h| h != current_hash)
            .unwrap_or(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_entry_creation() {
        let entry = ModuleEntry::new("npc/dialogue.json");
        assert_eq!(entry.path, "npc/dialogue.json");
        assert!(entry.hash.is_none());
        assert!(entry.tags.is_empty());
    }

    #[test]
    fn test_module_entry_with_hash_and_tags() {
        let entry = ModuleEntry::with_hash("npc/dialogue.json", "sha256:abc123")
            .with_tags(vec!["npc".to_string(), "dialogue".to_string()]);

        assert_eq!(entry.path, "npc/dialogue.json");
        assert_eq!(entry.hash, Some("sha256:abc123".to_string()));
        assert!(entry.has_tag("npc"));
        assert!(entry.has_tag("dialogue"));
        assert!(!entry.has_tag("quest"));
    }

    #[test]
    fn test_manifest_default() {
        let manifest = ModuleManifest::default();
        assert_eq!(manifest.version, "1.0");
        assert!(manifest.is_empty());
    }

    #[test]
    fn test_manifest_operations() {
        let mut manifest = ModuleManifest::new();

        manifest.add_module(
            "npc.dialogue",
            ModuleEntry::new("npc/dialogue.json").with_tags(vec!["npc".to_string()]),
        );

        manifest.add_module(
            "quest.generate",
            ModuleEntry::new("quest/generate.json").with_tags(vec!["quest".to_string()]),
        );

        assert_eq!(manifest.len(), 2);
        assert!(manifest.contains("npc.dialogue"));
        assert!(manifest.contains("quest.generate"));
        assert!(!manifest.contains("nonexistent"));

        let npc_modules = manifest.modules_with_tag("npc");
        assert_eq!(npc_modules.len(), 1);
        assert!(npc_modules.contains(&"npc.dialogue"));
    }

    #[test]
    fn test_manifest_deserialization() {
        let json = r#"{
            "version": "1.0",
            "modules": {
                "npc.dialogue.casual": {
                    "path": "npc/dialogue_casual.json",
                    "hash": "sha256:abc123",
                    "tags": ["npc", "dialogue"]
                },
                "quest.generate": {
                    "path": "quest/generate.json",
                    "tags": ["quest"]
                }
            }
        }"#;

        let manifest: ModuleManifest = serde_json::from_str(json).unwrap();

        assert_eq!(manifest.version, "1.0");
        assert_eq!(manifest.len(), 2);

        let npc = manifest.get("npc.dialogue.casual").unwrap();
        assert_eq!(npc.path, "npc/dialogue_casual.json");
        assert_eq!(npc.hash, Some("sha256:abc123".to_string()));
        assert!(npc.has_tag("npc"));

        let quest = manifest.get("quest.generate").unwrap();
        assert!(quest.hash.is_none()); // Optional field
    }

    #[test]
    fn test_manifest_serialization_roundtrip() {
        let mut manifest = ModuleManifest::with_version("1.0");
        manifest.add_module(
            "test.module",
            ModuleEntry::with_hash("test/module.json", "sha256:test123")
                .with_tags(vec!["test".to_string()]),
        );

        let json = serde_json::to_string(&manifest).unwrap();
        let deserialized: ModuleManifest = serde_json::from_str(&json).unwrap();

        assert_eq!(manifest.version, deserialized.version);
        assert_eq!(manifest.len(), deserialized.len());
        assert!(deserialized.contains("test.module"));
    }

    #[test]
    fn test_empty_manifest_handling() {
        let json = r#"{"version": "1.0"}"#;
        let manifest: ModuleManifest = serde_json::from_str(json).unwrap();

        assert_eq!(manifest.version, "1.0");
        assert!(manifest.is_empty());
    }

    #[test]
    fn test_manifest_remove() {
        let mut manifest = ModuleManifest::new();
        manifest.add_module("test", ModuleEntry::new("test.json"));

        assert!(manifest.contains("test"));
        let removed = manifest.remove("test");
        assert!(removed.is_some());
        assert!(!manifest.contains("test"));
    }
}
