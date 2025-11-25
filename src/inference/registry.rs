//! Signature Registry - Maps signature names to factory functions
//!
//! The SignatureRegistry allows consumers to register their DSPy signature types
//! at startup, which can then be referenced by name in module JSON files.
//!
//! # Design Pattern
//!
//! ml-crate-dsrs provides the registry infrastructure; consumers register their own signatures.
//! This decouples the module JSON format from the compiled signature types.
//!
//! # Example
//!
//! ```rust,ignore
//! use ml_crate_dsrs::inference::SignatureRegistry;
//! use dspy_rs::Signature;
//!
//! // Define your signatures
//! #[derive(Signature, Default)]
//! struct NPCDialogue {
//!     #[input] npc_personality: String,
//!     #[input] player_message: String,
//!     #[output] response: String,
//! }
//!
//! // Register them
//! let mut registry = SignatureRegistry::new();
//! registry.register::<NPCDialogue>("npc.dialogue");
//!
//! // Now modules can reference "npc.dialogue" by name
//! let signature = registry.create("npc.dialogue").unwrap();
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use dspy_rs::MetaSignature;

/// Factory function type for creating signature instances
///
/// The factory is a boxed function that creates a new boxed MetaSignature.
/// This allows us to store heterogeneous signature types in the same registry.
pub type SignatureFactory = Box<dyn Fn() -> Box<dyn MetaSignature> + Send + Sync>;

/// Registry mapping signature names to factory functions
///
/// The SignatureRegistry is the bridge between module JSON files (which reference
/// signatures by name) and compiled Rust signature types (which implement MetaSignature).
///
/// # Thread Safety
///
/// The registry itself is not thread-safe for mutation (registration).
/// Once built, it can be shared via Arc for concurrent reads.
pub struct SignatureRegistry {
    /// Map from signature name to factory function
    factories: HashMap<String, SignatureFactory>,
}

impl SignatureRegistry {
    /// Create an empty registry
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Register a signature type with a name
    ///
    /// The signature type must implement `MetaSignature` and `Default`.
    /// The `Default` implementation is used by the factory to create instances.
    ///
    /// # Arguments
    ///
    /// * `name` - The name to register the signature under (e.g., "npc.dialogue")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use dspy_rs::Signature;
    ///
    /// #[derive(Signature, Default)]
    /// struct MySignature {
    ///     #[input] question: String,
    ///     #[output] answer: String,
    /// }
    ///
    /// let mut registry = SignatureRegistry::new();
    /// registry.register::<MySignature>("my.signature");
    /// ```
    pub fn register<S>(&mut self, name: &str)
    where
        S: MetaSignature + Default + 'static,
    {
        self.factories.insert(
            name.to_string(),
            Box::new(|| Box::new(S::default()) as Box<dyn MetaSignature>),
        );
    }

    /// Register a signature with a custom factory function
    ///
    /// This allows more control over how signatures are created,
    /// for example with non-default initial values.
    ///
    /// # Arguments
    ///
    /// * `name` - The name to register the signature under
    /// * `factory` - A function that creates signature instances
    pub fn register_factory<F>(&mut self, name: &str, factory: F)
    where
        F: Fn() -> Box<dyn MetaSignature> + Send + Sync + 'static,
    {
        self.factories.insert(name.to_string(), Box::new(factory));
    }

    /// Create a signature instance by name
    ///
    /// Returns `None` if the signature name is not registered.
    ///
    /// # Arguments
    ///
    /// * `name` - The registered signature name
    ///
    /// # Returns
    ///
    /// A new boxed signature instance, or None if not found.
    pub fn create(&self, name: &str) -> Option<Box<dyn MetaSignature>> {
        self.factories.get(name).map(|factory| factory())
    }

    /// Check if a signature name is registered
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }

    /// List all registered signature names
    pub fn names(&self) -> Vec<&str> {
        self.factories.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of registered signatures
    pub fn len(&self) -> usize {
        self.factories.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.factories.is_empty()
    }

    /// Remove a signature from the registry
    pub fn unregister(&mut self, name: &str) -> bool {
        self.factories.remove(name).is_some()
    }

    /// Clear all registered signatures
    pub fn clear(&mut self) {
        self.factories.clear();
    }

    /// Wrap the registry in an Arc for sharing
    pub fn into_shared(self) -> Arc<Self> {
        Arc::new(self)
    }
}

impl Default for SignatureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SignatureRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SignatureRegistry")
            .field("signatures", &self.names())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspy_rs::Example;
    use serde_json::Value;

    /// Test signature for unit tests
    struct TestSignature {
        instruction: String,
        demos: Vec<Example>,
    }

    impl Default for TestSignature {
        fn default() -> Self {
            Self {
                instruction: "Test instruction".to_string(),
                demos: Vec::new(),
            }
        }
    }

    impl MetaSignature for TestSignature {
        fn demos(&self) -> Vec<Example> {
            self.demos.clone()
        }

        fn set_demos(&mut self, demos: Vec<Example>) -> anyhow::Result<()> {
            self.demos = demos;
            Ok(())
        }

        fn instruction(&self) -> String {
            self.instruction.clone()
        }

        fn input_fields(&self) -> Value {
            serde_json::json!(["question"])
        }

        fn output_fields(&self) -> Value {
            serde_json::json!(["answer"])
        }

        fn update_instruction(&mut self, instruction: String) -> anyhow::Result<()> {
            self.instruction = instruction;
            Ok(())
        }

        fn append(&mut self, _name: &str, _value: Value) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_registry_new() {
        let registry = SignatureRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_default() {
        let registry = SignatureRegistry::default();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_register_and_create() {
        let mut registry = SignatureRegistry::new();
        registry.register::<TestSignature>("test.signature");

        assert!(registry.contains("test.signature"));
        assert_eq!(registry.len(), 1);

        let signature = registry.create("test.signature");
        assert!(signature.is_some());

        let sig = signature.unwrap();
        assert_eq!(sig.instruction(), "Test instruction");
    }

    #[test]
    fn test_create_nonexistent_returns_none() {
        let registry = SignatureRegistry::new();
        assert!(registry.create("nonexistent").is_none());
    }

    #[test]
    fn test_register_multiple() {
        let mut registry = SignatureRegistry::new();
        registry.register::<TestSignature>("sig1");
        registry.register::<TestSignature>("sig2");
        registry.register::<TestSignature>("sig3");

        assert_eq!(registry.len(), 3);
        assert!(registry.contains("sig1"));
        assert!(registry.contains("sig2"));
        assert!(registry.contains("sig3"));
    }

    #[test]
    fn test_names() {
        let mut registry = SignatureRegistry::new();
        registry.register::<TestSignature>("alpha");
        registry.register::<TestSignature>("beta");

        let names = registry.names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
    }

    #[test]
    fn test_unregister() {
        let mut registry = SignatureRegistry::new();
        registry.register::<TestSignature>("test");

        assert!(registry.contains("test"));
        assert!(registry.unregister("test"));
        assert!(!registry.contains("test"));

        // Second unregister should return false
        assert!(!registry.unregister("test"));
    }

    #[test]
    fn test_clear() {
        let mut registry = SignatureRegistry::new();
        registry.register::<TestSignature>("sig1");
        registry.register::<TestSignature>("sig2");

        assert_eq!(registry.len(), 2);
        registry.clear();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_register_factory() {
        let mut registry = SignatureRegistry::new();

        registry.register_factory("custom", || {
            let mut sig = TestSignature::default();
            sig.instruction = "Custom instruction".to_string();
            Box::new(sig) as Box<dyn MetaSignature>
        });

        let signature = registry.create("custom").unwrap();
        assert_eq!(signature.instruction(), "Custom instruction");
    }

    #[test]
    fn test_create_returns_new_instance() {
        let mut registry = SignatureRegistry::new();
        registry.register::<TestSignature>("test");

        // Create two instances
        let sig1 = registry.create("test").unwrap();
        let sig2 = registry.create("test").unwrap();

        // They should be independent instances
        // (can't easily test this without mutation, but the factory should be called twice)
        assert_eq!(sig1.instruction(), sig2.instruction());
    }

    #[test]
    fn test_into_shared() {
        let mut registry = SignatureRegistry::new();
        registry.register::<TestSignature>("test");

        let shared = registry.into_shared();

        // Can still use the shared registry
        assert!(shared.contains("test"));
        assert!(shared.create("test").is_some());
    }

    #[test]
    fn test_debug_impl() {
        let mut registry = SignatureRegistry::new();
        registry.register::<TestSignature>("test.sig");

        let debug_str = format!("{:?}", registry);
        assert!(debug_str.contains("SignatureRegistry"));
        assert!(debug_str.contains("test.sig"));
    }

    #[test]
    fn test_overwrite_registration() {
        let mut registry = SignatureRegistry::new();

        registry.register_factory("test", || {
            let mut sig = TestSignature::default();
            sig.instruction = "First".to_string();
            Box::new(sig) as Box<dyn MetaSignature>
        });

        // Overwrite with new factory
        registry.register_factory("test", || {
            let mut sig = TestSignature::default();
            sig.instruction = "Second".to_string();
            Box::new(sig) as Box<dyn MetaSignature>
        });

        let sig = registry.create("test").unwrap();
        assert_eq!(sig.instruction(), "Second");
    }
}
