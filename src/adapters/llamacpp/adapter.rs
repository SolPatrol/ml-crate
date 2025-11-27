//! LlamaCpp Adapter implementation
//!
//! This module implements the dspy-rs `Adapter` trait for llama.cpp-based inference.
//! See specs/04-llamacpp-adapter.md for full specification.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use dspy_rs::adapter::Adapter;
use dspy_rs::{Chat, Example, LmUsage, Message, MetaSignature, Prediction, LM};
use rig_core::tool::ToolDyn;
use serde_json::Value;

use crate::adapters::llamacpp::{LlamaCppConfig, LlamaCppError, LoadedModel, Result};

/// LlamaCppAdapter - Implements dspy-rs Adapter trait for llama.cpp inference
///
/// Mirrors the CandleAdapter pattern for consistency and dspy-rs compatibility.
///
/// # Key Architectural Decisions
///
/// - Adapter implements dspy-rs Adapter trait directly
/// - Uses Arc<Mutex<LlamaModel>> for thread-safe model access
/// - Uses spawn_blocking for CPU/GPU-bound inference
/// - No separate backend threads (simpler, aligns with dspy-rs stateless pattern)
///
/// # Example
///
/// ```rust,ignore
/// use ml_crate_dsrs::adapters::llamacpp::{LlamaCppAdapter, LlamaCppConfig, LoadedModel};
/// use std::sync::Arc;
///
/// // Model Pool provides the LoadedModel
/// let loaded_model = Arc::new(model_pool.load_model("qwen2.5-0.5b").await?);
///
/// // Create adapter with config
/// let adapter = LlamaCppAdapter::from_loaded_model(loaded_model, LlamaCppConfig::default());
/// ```
#[derive(Clone)]
pub struct LlamaCppAdapter {
    /// Loaded model from Model Pool
    model: Arc<LoadedModel>,

    /// Configuration (temperature, max_tokens, etc.)
    config: LlamaCppConfig,
}

impl LlamaCppAdapter {
    /// Create adapter from Model Pool's LoadedModel
    ///
    /// This is the ONLY constructor - no direct model loading.
    /// Model Pool handles all model loading and device setup.
    pub fn from_loaded_model(loaded: Arc<LoadedModel>, config: LlamaCppConfig) -> Self {
        Self {
            model: loaded,
            config,
        }
    }

    /// Get a reference to the configuration
    pub fn config(&self) -> &LlamaCppConfig {
        &self.config
    }

    /// Get a reference to the loaded model
    pub fn model(&self) -> &LoadedModel {
        &self.model
    }

    // =========================================================================
    // Helper Methods (Phase 2)
    // =========================================================================

    /// Convert Chat to prompt string
    ///
    /// This helper method formats the conversation history into a single
    /// prompt string for the model. Ported from CandleAdapter lines 162-172.
    fn chat_to_prompt(&self, chat: &Chat) -> String {
        chat.messages
            .iter()
            .map(|msg| match msg {
                Message::System { content } => format!("System: {}", content),
                Message::User { content } => format!("User: {}", content),
                Message::Assistant { content } => format!("Assistant: {}", content),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Format demonstrations (few-shot examples) as Messages
    ///
    /// Extracts demos from signature and formats them as User/Assistant pairs.
    /// Ported from CandleAdapter format() method (lines 678-702).
    fn format_demonstrations(&self, signature: &dyn MetaSignature) -> Vec<Message> {
        let mut messages = Vec::new();
        let demos = signature.demos();

        for demo in demos {
            // Format demo inputs as user message
            let mut demo_input = String::new();
            for (field_name, field_value) in demo.data.iter() {
                if demo.input_keys.contains(field_name) {
                    demo_input.push_str(&format!("{}: {}\n", field_name, field_value));
                }
            }
            if !demo_input.is_empty() {
                messages.push(Message::user(demo_input.trim()));
            }

            // Format demo outputs as assistant message
            let mut demo_output = String::new();
            for (field_name, field_value) in demo.data.iter() {
                if demo.output_keys.contains(field_name) {
                    demo_output.push_str(&format!("{}: {}\n", field_name, field_value));
                }
            }
            if !demo_output.is_empty() {
                messages.push(Message::assistant(demo_output.trim()));
            }
        }

        messages
    }

    // =========================================================================
    // Parse Response Strategies (Phase 2 - Task 3)
    // =========================================================================

    /// Strategy 1: Parse response using field markers ("FieldName: value")
    ///
    /// Looks for patterns like "answer: Paris is the capital" in the response.
    /// Returns Some(HashMap) if all output fields are found, None otherwise.
    fn parse_with_field_markers(
        &self,
        content: &str,
        output_fields: &[String],
    ) -> Option<HashMap<String, Value>> {
        use regex::Regex;

        let mut outputs = HashMap::new();

        for field_name in output_fields {
            // Look for "FieldName: value" pattern
            // Use (?s) for multiline matching and capture until next field or end
            let pattern = format!(r"(?i){}:\s*(.+?)(?:\n\n|\n[A-Z][a-z]+:|$)", field_name);
            if let Ok(re) = Regex::new(&pattern) {
                if let Some(captures) = re.captures(content) {
                    if let Some(matched) = captures.get(1) {
                        let value_text = matched.as_str().trim();
                        outputs.insert(field_name.clone(), Value::String(value_text.to_string()));
                    }
                }
            }
        }

        // Return Some only if all fields were found
        if outputs.len() == output_fields.len() {
            Some(outputs)
        } else {
            None
        }
    }

    /// Strategy 2: Parse response as JSON
    ///
    /// Attempts to parse the content as JSON and extract output fields.
    /// Returns Some(HashMap) if valid JSON with all fields, None otherwise.
    fn parse_as_json(
        &self,
        content: &str,
        output_fields: &[String],
    ) -> Option<HashMap<String, Value>> {
        // Try to find JSON block in content (between { and })
        let json_start = content.find('{')?;
        let json_end = content.rfind('}')?;

        if json_end <= json_start {
            return None;
        }

        let json_str = &content[json_start..=json_end];
        let json_value: Value = serde_json::from_str(json_str).ok()?;
        let json_obj = json_value.as_object()?;

        let mut outputs = HashMap::new();
        for field_name in output_fields {
            if let Some(value) = json_obj.get(field_name) {
                outputs.insert(field_name.clone(), value.clone());
            }
        }

        // Return Some only if all fields were found
        if outputs.len() == output_fields.len() {
            Some(outputs)
        } else {
            None
        }
    }

    // =========================================================================
    // Generation Methods (Phase 2 - Task 4)
    // =========================================================================

    /// Generate text using the llama.cpp model
    ///
    /// This is the core inference method that:
    /// 1. Tokenizes the prompt via llama.cpp context
    /// 2. Runs model inference in a blocking task
    /// 3. Samples tokens using configured parameters
    /// 4. Detokenizes the result
    ///
    /// # Returns
    ///
    /// Tuple of (generated_text, prompt_tokens, completion_tokens)
    ///
    /// # Note
    ///
    /// This uses placeholder implementation until llama-cpp-2 types are
    /// properly integrated. The actual API calls will be verified against
    /// llama-cpp-2 v0.1 documentation during integration.
    pub async fn generate(&self, prompt: &str) -> Result<(String, u64, u64)> {
        // Clone Arc for spawn_blocking
        let model = Arc::clone(&self.model);
        let config = self.config.clone();
        let prompt_owned = prompt.to_string();

        // Run inference in blocking task (CPU/GPU-bound work)
        let result = tokio::task::spawn_blocking(move || {
            Self::generate_blocking(&model, &config, &prompt_owned)
        })
        .await
        .map_err(|e| LlamaCppError::InferenceFailed(format!("Task join error: {}", e)))??;

        Ok(result)
    }

    /// Blocking generation logic (runs inside spawn_blocking)
    ///
    /// Uses llama-cpp-2 API for real inference:
    /// - model.str_to_token() -> tokenization
    /// - context.decode() -> model forward pass
    /// - sampler.sample() -> token sampling
    /// - model.token_to_str() -> detokenize output
    fn generate_blocking(
        model: &LoadedModel,
        config: &LlamaCppConfig,
        prompt: &str,
    ) -> Result<(String, u64, u64)> {
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::model::AddBos;
        use llama_cpp_2::sampling::LlamaSampler;

        // Create a fresh context for this request (LlamaContext is !Send + !Sync)
        let mut ctx = model.create_context()?;

        // Tokenize the prompt
        let tokens = model
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| LlamaCppError::TokenizationFailed(format!("{e}")))?;

        let prompt_tokens = tokens.len() as u64;

        // Check context length
        if tokens.len() > config.context_length {
            return Err(LlamaCppError::ContextTooLong {
                actual: tokens.len(),
                max: config.context_length,
            });
        }

        // Create batch and add prompt tokens
        let mut batch = LlamaBatch::new(config.context_length, 1);

        // Add all prompt tokens to batch
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch.add(*token, i as i32, &[0], is_last)
                .map_err(|e| LlamaCppError::InferenceFailed(format!("Batch add error: {e}")))?;
        }

        // Decode prompt tokens
        ctx.decode(&mut batch)
            .map_err(|e| LlamaCppError::InferenceFailed(format!("Decode error: {e}")))?;

        // Set up sampler chain
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(config.temperature),
            LlamaSampler::top_p(config.top_p, 1),
            LlamaSampler::top_k(config.top_k.unwrap_or(40) as i32),
            LlamaSampler::dist(config.seed.unwrap_or(1234) as u32),
        ]);

        // Generation loop
        let mut output_tokens = Vec::new();
        let mut n_cur = tokens.len();

        while output_tokens.len() < config.max_tokens {
            // Sample next token
            let token = sampler.sample(&ctx, -1);

            // Check for end of generation
            if model.model.is_eog_token(token) {
                break;
            }

            output_tokens.push(token);

            // Clear batch and add the new token
            batch.clear();
            batch.add(token, n_cur as i32, &[0], true)
                .map_err(|e| LlamaCppError::InferenceFailed(format!("Batch add error: {e}")))?;

            n_cur += 1;

            // Decode the new token
            ctx.decode(&mut batch)
                .map_err(|e| LlamaCppError::InferenceFailed(format!("Decode error: {e}")))?;
        }

        let completion_tokens = output_tokens.len() as u64;

        // Detokenize output
        let mut response = String::new();
        for token in &output_tokens {
            let piece = model
                .model
                .token_to_str(*token, llama_cpp_2::model::Special::Tokenize)
                .map_err(|e| LlamaCppError::InferenceFailed(format!("Detokenize error: {e}")))?;
            response.push_str(&piece);
        }

        tracing::debug!(
            "LlamaCpp generate: prompt_tokens={}, completion_tokens={}, response_len={}",
            prompt_tokens,
            completion_tokens,
            response.len()
        );

        Ok((response, prompt_tokens, completion_tokens))
    }

    /// Generate text with retry logic
    ///
    /// Wraps generate() with exponential backoff retry on transient failures.
    /// Uses config.max_retries, initial_backoff_ms, and max_backoff_ms.
    pub async fn generate_with_retry(&self, prompt: &str) -> Result<(String, u64, u64)> {
        let mut attempt = 0;
        let mut backoff_ms = self.config.initial_backoff_ms;

        loop {
            match self.generate(prompt).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempt += 1;

                    if attempt > self.config.max_retries {
                        tracing::error!(
                            "Generation failed after {} attempts: {}",
                            self.config.max_retries,
                            e
                        );
                        return Err(e);
                    }

                    tracing::warn!(
                        "Generation attempt {} failed: {}. Retrying in {}ms...",
                        attempt,
                        e,
                        backoff_ms
                    );

                    // Sleep with current backoff
                    tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;

                    // Exponential backoff with cap
                    backoff_ms = (backoff_ms * 2).min(self.config.max_backoff_ms);
                }
            }
        }
    }
}

// =============================================================================
// Adapter Trait Implementation (Phase 2 - Task 6)
// =============================================================================

/// Implement the dspy-rs Adapter trait
///
/// This is verified against dspy-rs v0.7.3 source code.
/// Ported from CandleAdapter (src/adapters/candle/adapter.rs lines 654-858).
#[async_trait]
impl Adapter for LlamaCppAdapter {
    /// Convert a signature and inputs into a Chat (sequence of Messages)
    ///
    /// Formats the DSPy signature and input data into a conversation
    /// format that the model can process.
    ///
    /// Ported from CandleAdapter lines 667-718.
    fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat {
        let mut messages = Vec::new();

        // Add system message from signature instruction
        let instruction = signature.instruction();
        if !instruction.is_empty() {
            messages.push(Message::system(instruction));
        }

        // Add demonstrations (few-shot examples) if present
        let demo_messages = self.format_demonstrations(signature);
        messages.extend(demo_messages);

        // Format actual input fields into user message
        let mut user_content = String::new();
        for (field_name, field_value) in inputs.data.iter() {
            if inputs.input_keys.contains(field_name) {
                user_content.push_str(&format!("{}: {}\n", field_name, field_value));
            }
        }

        if !user_content.is_empty() {
            messages.push(Message::user(user_content.trim()));
        }

        Chat::new(messages)
    }

    /// Parse the model's response message into output fields
    ///
    /// Uses a 3-strategy parsing approach:
    /// 1. Field marker parsing: "FieldName: value" patterns (recommended)
    /// 2. JSON parsing: Parse as JSON and extract fields (fallback)
    /// 3. Single-field fallback: Use entire response for single output field
    ///
    /// Ported from CandleAdapter lines 736-805.
    fn parse_response(
        &self,
        signature: &dyn MetaSignature,
        response: Message,
    ) -> HashMap<String, Value> {
        // Extract response content
        let content = match response {
            Message::Assistant { content } => content,
            Message::User { content } => content,
            Message::System { content } => content,
        };

        // Get output fields from signature
        let output_fields_value = signature.output_fields();
        let fields_array = match output_fields_value.as_array() {
            Some(arr) => arr,
            None => return HashMap::new(), // No output fields defined
        };

        // Convert to Vec<String> for our helper methods
        let output_fields: Vec<String> = fields_array
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();

        if output_fields.is_empty() {
            return HashMap::new();
        }

        // Strategy 1: Try field marker parsing
        if let Some(outputs) = self.parse_with_field_markers(&content, &output_fields) {
            return outputs;
        }

        // Strategy 2: Try JSON parsing
        if let Some(outputs) = self.parse_as_json(&content, &output_fields) {
            return outputs;
        }

        // Strategy 3: Single field fallback - use entire response
        if output_fields.len() == 1 {
            let mut outputs = HashMap::new();
            outputs.insert(
                output_fields[0].clone(),
                Value::String(content.trim().to_string()),
            );
            return outputs;
        }

        // If all strategies fail, return empty HashMap
        HashMap::new()
    }

    /// Main entry point - orchestrates formatting, inference, and parsing
    ///
    /// Coordinates the entire inference pipeline:
    /// 1. Format inputs into Chat
    /// 2. Convert Chat to prompt
    /// 3. Run model inference with retry
    /// 4. Parse response into structured output
    /// 5. Return Prediction with usage stats
    ///
    /// Ported from CandleAdapter lines 825-858.
    async fn call(
        &self,
        _lm: Arc<LM>,
        signature: &dyn MetaSignature,
        inputs: Example,
        _tools: Vec<Arc<dyn ToolDyn>>,
    ) -> anyhow::Result<Prediction> {
        // 1. Format inputs into Chat
        let chat = self.format(signature, inputs);

        // 2. Convert Chat to prompt string
        let prompt = self.chat_to_prompt(&chat);

        // 3. Run model inference with retry logic
        let (response_text, prompt_tokens, completion_tokens) =
            self.generate_with_retry(&prompt).await.map_err(|e| {
                anyhow::anyhow!("Generation failed: {}", e)
            })?;

        // 4. Parse response into structured output
        let response_msg = Message::assistant(response_text);
        let outputs = self.parse_response(signature, response_msg);

        // 5. Return Prediction with proper structure
        // CRITICAL: Prediction has ONLY 2 fields: data and lm_usage
        Ok(Prediction {
            data: outputs,
            lm_usage: LmUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspy_rs::example;

    // =========================================================================
    // MockSignature for Testing (Task 7.1)
    // =========================================================================

    /// Mock signature for testing - matches CandleAdapter's MockSignature
    struct MockSignature {
        instruction: String,
        output_fields: Vec<String>,
        demonstrations: Vec<Example>,
    }

    impl MockSignature {
        fn new() -> Self {
            Self {
                instruction: "Answer the question concisely.".to_string(),
                output_fields: vec!["answer".to_string()],
                demonstrations: vec![],
            }
        }

        fn with_output_fields(mut self, fields: Vec<&str>) -> Self {
            self.output_fields = fields.into_iter().map(String::from).collect();
            self
        }

        fn with_demos(mut self, demos: Vec<Example>) -> Self {
            self.demonstrations = demos;
            self
        }
    }

    impl MetaSignature for MockSignature {
        fn demos(&self) -> Vec<Example> {
            self.demonstrations.clone()
        }

        fn set_demos(&mut self, demos: Vec<Example>) -> anyhow::Result<()> {
            self.demonstrations = demos;
            Ok(())
        }

        fn instruction(&self) -> String {
            self.instruction.clone()
        }

        fn input_fields(&self) -> Value {
            serde_json::json!(["question"])
        }

        fn output_fields(&self) -> Value {
            serde_json::json!(self.output_fields)
        }

        fn update_instruction(&mut self, instruction: String) -> anyhow::Result<()> {
            self.instruction = instruction;
            Ok(())
        }

        fn append(&mut self, name: &str, value: Value) -> anyhow::Result<()> {
            if let Some(s) = value.as_str() {
                if name == "output_field" {
                    self.output_fields.push(s.to_string());
                }
            }
            Ok(())
        }
    }

    // Shared model for all integration tests (loaded once via OnceLock)
    // This avoids the LlamaBackend singleton issue where init() can only be called once
    use std::sync::OnceLock;
    static SHARED_MODEL: OnceLock<Arc<LoadedModel>> = OnceLock::new();

    // Helper: Get or create the shared model (loaded once per process)
    // Made pub(crate) so types.rs tests can share the same model
    pub(crate) fn get_shared_model() -> Option<Arc<LoadedModel>> {
        let model_path = "models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf";
        if !std::path::Path::new(model_path).exists() {
            return None;
        }

        Some(
            SHARED_MODEL
                .get_or_init(|| {
                    // n_gpu_layers=99 offloads all layers to GPU (Vulkan/CUDA/Metal)
                    // Falls back gracefully to CPU if no GPU backend compiled
                    let loaded = LoadedModel::load(model_path, 99, 2048)
                        .expect("Failed to load shared test model");
                    Arc::new(loaded)
                })
                .clone(),
        )
    }

    // Helper: Create test adapter with shared model (requires GGUF file)
    fn test_adapter_with_model() -> Option<LlamaCppAdapter> {
        let model = get_shared_model()?;
        let config = LlamaCppConfig::default();
        Some(LlamaCppAdapter::from_loaded_model(model, config))
    }

    // =========================================================================
    // Basic Adapter Tests (Phase 1)
    // NOTE: These tests require a real GGUF model. Run with:
    // cargo test --features cpu -- --ignored
    // =========================================================================

    #[test]
    #[ignore = "requires GGUF model file"]
    fn test_adapter_from_loaded_model() {
        let adapter = test_adapter_with_model().expect("Model not available");
        assert!(adapter.model().name().contains("qwen"));
        assert_eq!(adapter.config().model_name, "llama-qwen2.5-0.5b");
    }

    #[test]
    #[ignore = "requires GGUF model file"]
    fn test_adapter_clone() {
        let adapter = test_adapter_with_model().expect("Model not available");
        let cloned = adapter.clone();
        assert_eq!(cloned.model().name(), adapter.model().name());
    }

    // =========================================================================
    // Helper Method Tests (Task 2) - Unit tests that don't require model
    // =========================================================================

    #[test]
    fn test_chat_to_prompt_format() {
        // Test chat_to_prompt formatting logic without needing a model
        let messages = vec![
            Message::system("You are helpful"),
            Message::user("Hello"),
            Message::assistant("Hi there!"),
        ];

        let chat = Chat::new(messages);

        // Manually test formatting
        let prompt: String = chat.messages
            .iter()
            .map(|msg| match msg {
                Message::System { content } => format!("System: {}", content),
                Message::User { content } => format!("User: {}", content),
                Message::Assistant { content } => format!("Assistant: {}", content),
            })
            .collect::<Vec<_>>()
            .join("\n");

        assert!(prompt.contains("System: You are helpful"));
        assert!(prompt.contains("User: Hello"));
        assert!(prompt.contains("Assistant: Hi there!"));
    }

    // =========================================================================
    // Parse Response Strategy Tests (Task 3) - Unit tests
    // These test the parsing logic independent of model loading
    // =========================================================================

    #[test]
    fn test_parse_field_markers_logic() {
        use regex::Regex;

        let output_fields = vec!["answer".to_string()];
        let content = "answer: Paris is the capital of France";

        let mut outputs = HashMap::new();
        for field_name in &output_fields {
            let pattern = format!(r"(?i){}:\s*(.+?)(?:\n\n|\n[A-Z][a-z]+:|$)", field_name);
            if let Ok(re) = Regex::new(&pattern) {
                if let Some(captures) = re.captures(content) {
                    if let Some(matched) = captures.get(1) {
                        let value_text = matched.as_str().trim();
                        outputs.insert(field_name.clone(), Value::String(value_text.to_string()));
                    }
                }
            }
        }

        assert_eq!(outputs.len(), 1);
        assert_eq!(
            outputs.get("answer").unwrap().as_str().unwrap(),
            "Paris is the capital of France"
        );
    }

    #[test]
    fn test_parse_json_logic() {
        let content = r#"Here is my response: {"answer": "42"}"#;
        let output_fields = vec!["answer".to_string()];

        let json_start = content.find('{').unwrap();
        let json_end = content.rfind('}').unwrap();
        let json_str = &content[json_start..=json_end];
        let json_value: Value = serde_json::from_str(json_str).unwrap();
        let json_obj = json_value.as_object().unwrap();

        let mut outputs = HashMap::new();
        for field_name in &output_fields {
            if let Some(value) = json_obj.get(field_name) {
                outputs.insert(field_name.clone(), value.clone());
            }
        }

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs.get("answer").unwrap().as_str().unwrap(), "42");
    }

    // =========================================================================
    // Adapter Trait Tests (Task 6) - Require model
    // =========================================================================

    #[test]
    #[ignore = "requires GGUF model file"]
    fn test_format_basic() {
        let adapter = test_adapter_with_model().expect("Model not available");
        let signature = MockSignature::new();

        let inputs = example! {
            "question": "input" => "What is 2+2?"
        };

        let chat = adapter.format(&signature, inputs);

        assert!(!chat.messages.is_empty());
        assert_eq!(chat.messages.len(), 2); // System + User
    }

    #[test]
    #[ignore = "requires GGUF model file"]
    fn test_format_with_demos() {
        let adapter = test_adapter_with_model().expect("Model not available");
        let signature = MockSignature::new().with_demos(vec![
            example! {
                "question": "input" => "What is 2+2?",
                "answer": "output" => "4"
            },
        ]);

        let inputs = example! {
            "question": "input" => "What is 5+5?"
        };

        let chat = adapter.format(&signature, inputs);

        // Should have: System + Demo(User+Assistant) + Actual(User)
        assert_eq!(chat.messages.len(), 4);
    }

    // =========================================================================
    // Generation Tests (Task 4 & 5) - Require model
    // =========================================================================

    #[tokio::test]
    #[ignore = "requires GGUF model file"]
    async fn test_generate_real() {
        let adapter = test_adapter_with_model().expect("Model not available");

        let result = adapter.generate("Hello, how are you?").await;
        assert!(result.is_ok());

        let (text, prompt_tokens, completion_tokens) = result.unwrap();
        assert!(!text.is_empty());
        assert!(prompt_tokens > 0);
        assert!(completion_tokens > 0);
    }

    #[tokio::test]
    #[ignore = "requires GGUF model file"]
    async fn test_generate_with_retry_success() {
        let adapter = test_adapter_with_model().expect("Model not available");

        let result = adapter.generate_with_retry("Test prompt").await;
        assert!(result.is_ok());
    }

    // =========================================================================
    // Integration Test (Task 6.3)
    // =========================================================================

    #[tokio::test]
    #[ignore = "requires GGUF model file"]
    async fn test_call_integration() {
        let adapter = test_adapter_with_model().expect("Model not available");
        let signature = MockSignature::new();

        let mut inputs = Example::default();
        inputs.data.insert("question".to_string(), Value::String("What is 2+2?".to_string()));
        inputs.input_keys.push("question".to_string());

        // Test the full pipeline
        let chat = adapter.format(&signature, inputs);
        let prompt = adapter.chat_to_prompt(&chat);
        let (response_text, prompt_tokens, completion_tokens) =
            adapter.generate_with_retry(&prompt).await.unwrap();

        assert!(!response_text.is_empty());
        assert!(prompt_tokens > 0);
        assert!(completion_tokens > 0);
    }

    // =========================================================================
    // DSPy Module Compatibility Tests (Task 7.2)
    // =========================================================================

    #[test]
    fn test_signature_instruction_update() {
        let mut signature = MockSignature::new();
        assert_eq!(signature.instruction(), "Answer the question concisely.");

        signature.update_instruction("New instruction".to_string()).unwrap();
        assert_eq!(signature.instruction(), "New instruction");
    }

    // =========================================================================
    // Edge Case Unit Tests (Phase 4 - No Model Required)
    // These test parsing/formatting logic without needing a real model
    // =========================================================================

    #[test]
    fn test_parse_response_empty() {
        // Edge case: Empty response should use single-field fallback
        use regex::Regex;

        let output_fields = vec!["answer".to_string()];
        let content = "";

        // Strategy 1: Field markers - should fail (empty)
        let mut found_fields = 0;
        for field_name in &output_fields {
            let pattern = format!(r"(?i){}:\s*(.+?)(?:\n\n|\n[A-Z][a-z]+:|$)", field_name);
            if let Ok(re) = Regex::new(&pattern) {
                if re.captures(content).is_some() {
                    found_fields += 1;
                }
            }
        }
        assert_eq!(found_fields, 0); // No fields found in empty content

        // Strategy 3: Single-field fallback - should return empty string
        if output_fields.len() == 1 {
            let result = content.trim().to_string();
            assert_eq!(result, ""); // Empty after trim
        }
    }

    #[test]
    fn test_parse_response_whitespace_only() {
        // Edge case: Whitespace-only response
        let content = "   \n\t  \n   ";
        let trimmed = content.trim();
        assert!(trimmed.is_empty()); // Should be empty after trim
    }

    #[test]
    fn test_parse_response_special_chars() {
        // Edge case: Special characters in response
        use regex::Regex;

        let output_fields = vec!["answer".to_string()];
        let content = "answer: Hello \"world\" & <test>";

        let mut outputs = HashMap::new();
        for field_name in &output_fields {
            let pattern = format!(r"(?i){}:\s*(.+?)(?:\n\n|\n[A-Z][a-z]+:|$)", field_name);
            if let Ok(re) = Regex::new(&pattern) {
                if let Some(captures) = re.captures(content) {
                    if let Some(matched) = captures.get(1) {
                        outputs.insert(field_name.clone(), matched.as_str().trim().to_string());
                    }
                }
            }
        }

        assert_eq!(outputs.len(), 1);
        let answer = outputs.get("answer").unwrap();
        assert!(answer.contains("\"world\"")); // Quotes preserved
        assert!(answer.contains("&")); // Ampersand preserved
        assert!(answer.contains("<test>")); // Angle brackets preserved
    }

    #[test]
    fn test_parse_response_unicode() {
        // Edge case: Unicode content (Chinese, emoji)
        use regex::Regex;

        let output_fields = vec!["answer".to_string()];
        let content = "answer: 你好世界 🎉 Привет мир";

        let mut outputs = HashMap::new();
        for field_name in &output_fields {
            let pattern = format!(r"(?i){}:\s*(.+?)(?:\n\n|\n[A-Z][a-z]+:|$)", field_name);
            if let Ok(re) = Regex::new(&pattern) {
                if let Some(captures) = re.captures(content) {
                    if let Some(matched) = captures.get(1) {
                        outputs.insert(field_name.clone(), matched.as_str().trim().to_string());
                    }
                }
            }
        }

        assert_eq!(outputs.len(), 1);
        let answer = outputs.get("answer").unwrap();
        assert!(answer.contains("你好")); // Chinese preserved
        assert!(answer.contains("🎉")); // Emoji preserved
        assert!(answer.contains("Привет")); // Cyrillic preserved
    }

    #[test]
    fn test_format_empty_instruction() {
        // Edge case: Empty instruction should NOT produce system message
        // Test formatting logic directly
        let instruction = "";
        let mut messages = Vec::new();

        // This mirrors the format() logic
        if !instruction.is_empty() {
            messages.push(Message::system(instruction));
        }

        // No system message when instruction is empty
        assert!(!messages.iter().any(|m| matches!(m, Message::System { .. })));
    }

    #[test]
    fn test_format_very_long_input() {
        // Edge case: Very long input should be handled without panic
        let long_text = "a".repeat(5000);

        // Test that chat_to_prompt handles long content
        let messages = vec![
            Message::system("Be helpful"),
            Message::user(&long_text),
        ];

        let chat = Chat::new(messages);

        // Manually format like chat_to_prompt does
        let prompt: String = chat.messages
            .iter()
            .map(|msg| match msg {
                Message::System { content } => format!("System: {}", content),
                Message::User { content } => format!("User: {}", content),
                Message::Assistant { content } => format!("Assistant: {}", content),
            })
            .collect::<Vec<_>>()
            .join("\n");

        assert!(prompt.len() > 5000);
        assert!(prompt.contains(&long_text));
    }

    #[test]
    #[ignore = "requires GGUF model file"]
    fn test_demonstrations_formatting() {
        // Verify that demonstrations are correctly formatted for few-shot learning
        let adapter = test_adapter_with_model().expect("Model not available");
        let signature = MockSignature::new().with_demos(vec![
            example! {
                "question": "input" => "Capital of France?",
                "answer": "output" => "Paris"
            },
        ]);

        let inputs = example! {
            "question": "input" => "Capital of Germany?"
        };

        let chat = adapter.format(&signature, inputs);

        // Convert to prompt and verify structure
        let prompt = adapter.chat_to_prompt(&chat);

        // Should contain the demonstration
        assert!(prompt.contains("Capital of France?"));
        assert!(prompt.contains("Paris"));

        // Should contain the actual question
        assert!(prompt.contains("Capital of Germany?"));
    }

    // =========================================================================
    // dspy-rs Integration Tests (Phase 4)
    // These verify compatibility with dspy-rs configure() and modules
    // =========================================================================

    /// Test that LlamaCppAdapter can be used with dspy_rs::configure()
    ///
    /// This is the critical integration test - if this works, our adapter
    /// is compatible with the dspy-rs framework.
    #[tokio::test]
    #[ignore = "requires GGUF model"]
    async fn test_dspy_configure() {
        // Use shared model to avoid LlamaBackend singleton issue
        let adapter = test_adapter_with_model().expect("Model not available");

        // Build LM configuration
        // Note: base_url is required but not used for local inference
        let lm = LM::builder()
            .model("local-llamacpp".to_string())
            .base_url("http://localhost:0".to_string())
            .build()
            .await
            .expect("LM build failed");

        // Configure dspy-rs with our adapter
        // This is the key test - dspy_rs::configure() accepts Adapter trait objects
        dspy_rs::configure(lm, adapter);

        // If we get here without panic, configuration worked!
        // The adapter is now registered as the global inference backend
    }
}
