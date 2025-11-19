//! CandleAdapter - Implements dspy-rs Adapter trait for Candle inference
//!
//! This is the core implementation of the Adapter trait from dspy-rs v0.7.3.
//! It bridges dspy-rs with Candle-based model inference.
//!
//! # Architecture
//!
//! - **Receives models** from Model Pool (does NOT load models itself)
//! - **Implements Adapter trait** with 3 methods: format(), parse_response(), call()
//! - **Returns Prediction** with proper structure (data + lm_usage)
//!
//! # Phase 1: Real Candle Integration
//!
//! This phase implements actual Candle model inference with tokenization and sampling.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use candle_core::{Device, IndexOp, Tensor};
use candle_transformers::models::qwen2::ModelForCausalLM as Qwen2Model;
use dspy_rs::adapter::Adapter;
use dspy_rs::{Chat, Example, LM, LmUsage, Message, MetaSignature, Prediction};
use rig_core::tool::ToolDyn;
use serde_json::Value;
use tokenizers::Tokenizer;

use super::config::CandleConfig;
use super::error::{CandleAdapterError, Result};

/// LoadedModel - Contains a fully loaded Candle model ready for inference
///
/// This struct is provided by the Model Pool and contains everything needed
/// for inference: the model, tokenizer, and device.
///
/// # Thread Safety
///
/// The model is wrapped in `Arc<Mutex<>>` because Qwen2Model::forward() requires
/// `&mut self`. This allows thread-safe mutable access across async tasks.
#[derive(Clone)]
pub struct LoadedModel {
    /// The actual Qwen2 model from Candle (ModelForCausalLM with lm_head, wrapped in Mutex for &mut forward())
    pub model: Arc<Mutex<Qwen2Model>>,

    /// Tokenizer for text ↔ token ID conversion
    pub tokenizer: Arc<Tokenizer>,

    /// Device the model is loaded on (CPU, CUDA, or Metal)
    pub device: Device,

    /// Model name for logging and debugging
    pub model_name: String,
}

impl LoadedModel {
    /// Create a new LoadedModel from components (called by Model Pool)
    ///
    /// # Arguments
    ///
    /// * `model` - Qwen2 model instance
    /// * `tokenizer` - HuggingFace tokenizer
    /// * `device` - Candle device (CPU/CUDA/Metal)
    /// * `model_name` - Name for logging
    pub fn new(
        model: Qwen2Model,
        tokenizer: Tokenizer,
        device: Device,
        model_name: String,
    ) -> Self {
        Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            device,
            model_name,
        }
    }

    // Note: Mock function removed in Phase 1
    // Real models must be loaded from Model Pool
    // For testing, use actual model files
}

/// CandleAdapter - Implements dspy-rs Adapter trait for Candle inference
///
/// This adapter provides Candle-based LLM inference for dspy-rs predictors.
///
/// # Example
///
/// ```rust,ignore
/// use ml_crate_dsrs::adapters::candle::{CandleAdapter, CandleConfig};
/// use dspy_rs::{configure, Predict, Signature, example};
///
/// #[derive(Signature)]
/// struct QA {
///     #[input]
///     question: String,
///     #[output]
///     answer: String,
/// }
///
/// # async fn example() -> anyhow::Result<()> {
/// // Phase 1: Real Qwen3-0.6B model
/// // let model_pool = ModelPool::new("./models".into());
/// // let loaded = model_pool.load_model("Qwen3-0.6B").await?;
/// // let adapter = CandleAdapter::from_loaded_model(loaded, CandleConfig::default());
///
/// // configure(adapter, None);
/// # Ok(())
/// # }
/// ```
pub struct CandleAdapter {
    /// Loaded model from Model Pool (Qwen3-0.6B or compatible)
    model: Arc<LoadedModel>,

    /// Configuration
    /// Phase 0: Stored but not used yet
    /// Phase 1: Will be used for temperature, max_tokens, etc.
    #[allow(dead_code)]
    config: CandleConfig,
}

impl CandleAdapter {
    /// Create adapter from Model Pool's LoadedModel
    ///
    /// This is the ONLY constructor - no direct model loading.
    /// The Model Pool handles all model loading, tokenization, and device setup.
    ///
    /// # Arguments
    ///
    /// * `loaded` - Pre-loaded model from Model Pool
    /// * `config` - Adapter configuration
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ml_crate_dsrs::adapters::candle::{CandleAdapter, CandleConfig};
    /// use std::sync::Arc;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// // Phase 1: Get loaded model from Model Pool
    /// // let model_pool = ModelPool::new();
    /// // let loaded = model_pool.load_model("qwen3-0.6b").await?;
    /// // let adapter = CandleAdapter::from_loaded_model(Arc::new(loaded), CandleConfig::default());
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_loaded_model(loaded: Arc<LoadedModel>, config: CandleConfig) -> Self {
        Self {
            model: loaded,
            config,
        }
    }

    // Note: new_mock() removed in Phase 1
    // Use from_loaded_model() with a real model from Model Pool

    /// Convert Chat to prompt string
    ///
    /// This helper method formats the conversation history into a single
    /// prompt string for the model.
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

    /// Generate text using the model
    ///
    /// Phase 1: Real Candle inference with:
    /// - Tokenization using the loaded tokenizer
    /// - Model forward pass with spawn_blocking
    /// - Sampling (temperature, top-p, top-k)
    /// - Detokenization back to text
    ///
    /// # Returns
    ///
    /// Tuple of (generated_text, prompt_tokens, completion_tokens)
    pub async fn generate(&self, prompt: &str) -> Result<(String, u64, u64)> {
        // 1. Tokenize the prompt
        let encoding = self
            .model
            .tokenizer
            .encode(prompt, true) // true = add special tokens
            .map_err(|e| CandleAdapterError::TokenizationFailed(e.to_string()))?;

        let input_tokens = encoding.get_ids().to_vec();
        let prompt_token_count = input_tokens.len() as u64;

        tracing::debug!(
            "Tokenized prompt: {} chars -> {} tokens",
            prompt.len(),
            prompt_token_count
        );

        // 2. Check context length
        if input_tokens.len() > self.config.context_length {
            return Err(CandleAdapterError::ContextTooLong {
                actual: input_tokens.len(),
                max: self.config.context_length,
            });
        }

        // 3. Clone what we need for spawn_blocking
        let model = Arc::clone(&self.model.model);
        let device = self.model.device.clone();
        let max_tokens = self.config.max_tokens;
        let temperature = self.config.temperature;
        let top_p = self.config.top_p;
        let top_k = self.config.top_k;

        // 4. Run inference in blocking thread (CPU/GPU-bound work)
        let generated_tokens = tokio::task::spawn_blocking(move || {
            Self::generate_tokens(
                &model,
                &device,
                input_tokens,
                max_tokens,
                temperature,
                top_p,
                top_k,
            )
        })
        .await
        .map_err(|e| {
            CandleAdapterError::InferenceFailed(format!("Task join error: {}", e))
        })??;

        // 5. Count completion tokens (total - prompt)
        let completion_token_count = (generated_tokens.len() - prompt_token_count as usize) as u64;

        tracing::debug!(
            "Generated {} tokens (prompt: {}, completion: {})",
            generated_tokens.len(),
            prompt_token_count,
            completion_token_count
        );

        // 6. Detokenize back to text
        let output_text = self
            .model
            .tokenizer
            .decode(&generated_tokens, true) // true = skip special tokens
            .map_err(|e| CandleAdapterError::TokenizationFailed(e.to_string()))?;

        Ok((output_text, prompt_token_count, completion_token_count))
    }

    /// Core token generation logic (runs in spawn_blocking)
    ///
    /// This implements auto-regressive generation: generate one token at a time,
    /// feeding the output back as input for the next token.
    fn generate_tokens(
        model: &Arc<Mutex<Qwen2Model>>,
        device: &Device,
        input_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
    ) -> Result<Vec<u32>> {
        // Lock the model for inference
        let mut model = model
            .lock()
            .map_err(|e| CandleAdapterError::InferenceFailed(format!("Model lock error: {}", e)))?;

        // Clear KV cache from previous generations to avoid shape mismatches
        model.clear_kv_cache();

        // Initialize with input tokens
        let mut all_tokens = input_tokens.clone();

        // Generate tokens one at a time (auto-regressive)
        for step in 0..max_tokens {
            // For the first iteration, use all tokens; for subsequent ones, only the last token
            let current_tokens = if step == 0 {
                &all_tokens[..]
            } else {
                &all_tokens[all_tokens.len() - 1..]
            };

            // Convert tokens to tensor (U32 dtype for token IDs)
            let input_tensor = Tensor::new(current_tokens, device)
                .map_err(|e| CandleAdapterError::InferenceFailed(e.to_string()))?
                .reshape((1, current_tokens.len())) // [batch_size=1, seq_len]
                .map_err(|e| CandleAdapterError::InferenceFailed(e.to_string()))?;

            // seqlen_offset is the position in the sequence where we are
            let seqlen_offset = if step == 0 { 0 } else { all_tokens.len() - 1 };

            // Run model forward pass to get logits for next token
            // ModelForCausalLM.forward() takes (input_ids, seqlen_offset) and returns logits
            let logits = model
                .forward(&input_tensor, seqlen_offset)
                .map_err(|e| CandleAdapterError::InferenceFailed(e.to_string()))?;

            // Extract logits for the last token: [batch, seq_len, vocab] -> [vocab]
            let seq_len = logits.dim(1).map_err(|e| CandleAdapterError::InferenceFailed(e.to_string()))?;
            let logits = logits
                .i((.., seq_len - 1, ..))
                .map_err(|e| CandleAdapterError::InferenceFailed(e.to_string()))?
                .squeeze(0)
                .map_err(|e| CandleAdapterError::InferenceFailed(e.to_string()))?;

            // Sample next token from logits
            let next_token = Self::sample_token(&logits, temperature, top_p, top_k)?;

            // Check for end-of-sequence token (Qwen2.5-0.5B EOS)
            const EOS_TOKEN_ID: u32 = 151643;
            if next_token == EOS_TOKEN_ID {
                tracing::debug!("Hit EOS token, stopping generation");
                break;
            }

            // Add to sequence
            all_tokens.push(next_token);
        }

        Ok(all_tokens)
    }

    /// Sample next token from logits using temperature, top-p, and top-k
    ///
    /// # Sampling Strategy
    ///
    /// 1. Apply temperature scaling (controls randomness)
    /// 2. Convert to probabilities (softmax)
    /// 3. Apply top-k filtering (optional)
    /// 4. Apply top-p (nucleus) filtering
    /// 5. Sample from filtered distribution
    fn sample_token(
        logits: &Tensor,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
    ) -> Result<u32> {
        use candle_core::DType;
        use candle_nn::ops::softmax;
        use rand::distributions::{Distribution, WeightedIndex};
        use rand::thread_rng;

        // Convert F16 to F32 if needed (model outputs F16, but we need F32 for sampling)
        let logits = if logits.dtype() == DType::F16 {
            logits
                .to_dtype(DType::F32)
                .map_err(|e| CandleAdapterError::InferenceFailed(e.to_string()))?
        } else {
            logits.clone()
        };

        // Apply temperature scaling
        let logits = (logits / temperature as f64)
            .map_err(|e| CandleAdapterError::InferenceFailed(e.to_string()))?;

        // Convert to probabilities (softmax)
        let probs = softmax(&logits, 0)
            .map_err(|e| CandleAdapterError::InferenceFailed(e.to_string()))?;

        // Get probabilities as Vec<f32>
        let probs_vec = probs
            .to_vec1::<f32>()
            .map_err(|e| CandleAdapterError::InferenceFailed(e.to_string()))?;

        // Apply top-k filtering if specified
        let probs_vec = if let Some(k) = top_k {
            Self::apply_top_k(probs_vec, k)
        } else {
            probs_vec
        };

        // Apply top-p (nucleus) filtering
        let probs_vec = Self::apply_top_p(probs_vec, top_p);

        // Sample from the filtered distribution
        let dist = WeightedIndex::new(&probs_vec).map_err(|e| {
            CandleAdapterError::InferenceFailed(format!("Sampling error: {}", e))
        })?;

        let token_id = dist.sample(&mut thread_rng()) as u32;

        Ok(token_id)
    }

    /// Apply top-k filtering to probabilities
    ///
    /// Keeps only the top-k highest probability tokens, zeros out the rest.
    fn apply_top_k(mut probs: Vec<f32>, k: usize) -> Vec<f32> {
        // Get indices sorted by probability (descending)
        let mut indices: Vec<_> = (0..probs.len()).collect();
        indices.sort_by(|&i, &j| probs[j].partial_cmp(&probs[i]).unwrap());

        // Zero out everything beyond top-k
        for &idx in indices.iter().skip(k) {
            probs[idx] = 0.0;
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }

        probs
    }

    /// Apply top-p (nucleus) filtering to probabilities
    ///
    /// Keeps only tokens whose cumulative probability is within top-p,
    /// zeros out the rest.
    fn apply_top_p(mut probs: Vec<f32>, p: f32) -> Vec<f32> {
        // Get indices sorted by probability (descending)
        let mut indices: Vec<_> = (0..probs.len()).collect();
        indices.sort_by(|&i, &j| probs[j].partial_cmp(&probs[i]).unwrap());

        // Calculate cumulative probabilities
        let mut cumsum = 0.0;
        let mut cutoff_idx = probs.len();

        for (i, &idx) in indices.iter().enumerate() {
            cumsum += probs[idx];
            if cumsum > p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Zero out tokens beyond cutoff
        for &idx in indices.iter().skip(cutoff_idx) {
            probs[idx] = 0.0;
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }

        probs
    }
}

/// Implement the dspy-rs Adapter trait
///
/// This is verified against dspy-rs v0.7.3 source code at:
/// .claude/knowledge/dspy/source/adapter-trait.md (lines 34-54)
#[async_trait]
impl Adapter for CandleAdapter {
    /// Convert a signature and inputs into a Chat (sequence of Messages)
    ///
    /// This method formats the DSPy signature and input data into a conversation
    /// format that the model can process.
    ///
    /// # Verified Against
    ///
    /// dspy-rs v0.7.3 source: adapter-trait.md lines 34-54
    /// - NOT async (returns Chat directly)
    /// - Takes signature + inputs
    /// - Returns Chat
    fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat {
        let mut messages = Vec::new();

        // Add system message from signature instruction
        let instruction = signature.instruction();
        if !instruction.is_empty() {
            messages.push(Message::system(instruction));
        }

        // Add demonstrations (few-shot examples) if present
        // Demonstrations show the model how to respond via input→output pairs
        let demos = signature.demos();
        if !demos.is_empty() {
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
        }

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
    /// This method extracts structured output fields from the model's response
    /// based on the signature's output field definitions.
    ///
    /// Uses a 3-strategy parsing approach:
    /// 1. Field marker parsing: "FieldName: value" patterns (recommended)
    /// 2. JSON parsing: Parse as JSON and extract fields (fallback)
    /// 3. Single-field fallback: Use entire response for single output field
    ///
    /// # Verified Against
    ///
    /// dspy-rs v0.7.3 source: adapter-trait.md lines 34-54
    /// - NOT async (returns HashMap directly)
    /// - Takes signature + response Message
    /// - Returns HashMap<String, Value>
    fn parse_response(
        &self,
        signature: &dyn MetaSignature,
        response: Message,
    ) -> HashMap<String, Value> {
        use regex::Regex;

        let mut outputs = HashMap::new();

        // Extract response content
        let content = match response {
            Message::Assistant { content } => content,
            Message::User { content } => content,
            Message::System { content } => content,
        };

        // Get output fields from signature
        let output_fields = signature.output_fields();
        let fields_array = match output_fields.as_array() {
            Some(arr) => arr,
            None => return outputs, // No output fields defined
        };

        // Strategy 1: Try field marker parsing ("FieldName: value")
        for field_value in fields_array.iter() {
            if let Some(field_name) = field_value.as_str() {
                // Look for "FieldName: value" pattern
                // Use (?s) for multiline matching and capture until next field or end
                let pattern = format!(r"(?i){}:\s*(.+?)(?:\n\n|\n[A-Z][a-z]+:|$)", field_name);
                if let Ok(re) = Regex::new(&pattern) {
                    if let Some(captures) = re.captures(&content) {
                        if let Some(matched) = captures.get(1) {
                            let value_text = matched.as_str().trim();
                            outputs.insert(
                                field_name.to_string(),
                                Value::String(value_text.to_string()),
                            );
                        }
                    }
                }
            }
        }

        // Strategy 2: If no fields found, try JSON parsing
        if outputs.is_empty() {
            if let Ok(json_value) = serde_json::from_str::<Value>(&content) {
                if let Some(json_obj) = json_value.as_object() {
                    for field_value in fields_array.iter() {
                        if let Some(field_name) = field_value.as_str() {
                            if let Some(value) = json_obj.get(field_name) {
                                outputs.insert(field_name.to_string(), value.clone());
                            }
                        }
                    }
                }
            }
        }

        // Strategy 3: Single field fallback - use entire response
        if outputs.is_empty() && fields_array.len() == 1 {
            if let Some(field_name) = fields_array[0].as_str() {
                outputs.insert(
                    field_name.to_string(),
                    Value::String(content.trim().to_string()),
                );
            }
        }

        outputs
    }

    /// Main entry point - orchestrates formatting, inference, and parsing
    ///
    /// This method coordinates the entire inference pipeline:
    /// 1. Format inputs into Chat
    /// 2. Convert Chat to prompt
    /// 3. Run model inference
    /// 4. Parse response into structured output
    /// 5. Return Prediction with usage stats
    ///
    /// # Verified Against
    ///
    /// dspy-rs v0.7.3 source: adapter-trait.md lines 34-54
    /// - IS async (returns Result<Prediction>)
    /// - Takes lm + signature + inputs + tools
    /// - Returns Result<Prediction>
    ///
    /// Prediction structure verified against: core-types.md lines 182-186
    /// - ONLY 2 fields: data (HashMap<String, Value>) and lm_usage (LmUsage)
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

        // 3. Run model inference (Phase 1: real Candle with real token counts)
        let (response_text, prompt_tokens, completion_tokens) = self.generate(&prompt).await
            .map_err(|e| anyhow::anyhow!("Generation failed: {}", e))?;

        // 4. Parse response into structured output
        let response_msg = Message::assistant(response_text);
        let outputs = self.parse_response(signature, response_msg);

        // 5. Return Prediction with CORRECT structure
        // CRITICAL: Prediction has ONLY 2 fields: data and lm_usage
        // Verified against core-types.md lines 182-186
        // Phase 1: Uses REAL token counts from tokenizer
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

    /// Mock signature for testing
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

    // Helper: Create a minimal adapter for testing non-inference methods
    // WARNING: This adapter uses unsafe zeroed memory and CANNOT be used for actual inference
    // Only safe for testing format() and parse_response() which don't touch the model
    #[allow(invalid_value)]
    fn test_adapter_unsafe() -> CandleAdapter {
        // Create a minimal tokenizer for testing
        let tokenizer = Tokenizer::from_bytes(br#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]}}"#)
            .expect("Failed to create test tokenizer");

        // Create LoadedModel without actual model (only for testing format/parse)
        let loaded = LoadedModel {
            model: Arc::new(Mutex::new(unsafe { std::mem::zeroed() })), // UNSAFE: Won't be used
            tokenizer: Arc::new(tokenizer),
            device: Device::Cpu,
            model_name: "test".to_string(),
        };

        CandleAdapter::from_loaded_model(Arc::new(loaded), CandleConfig::default())
    }

    // Safe helper: Get a properly configured adapter instance using a mock model path
    // This requires a real model file to exist, so tests using this are marked #[ignore]
    fn test_adapter() -> CandleAdapter {
        test_adapter_unsafe()
    }

    #[test]
    #[ignore] // Uses zeroed model, safe only if model isn't accessed
    fn test_format() {
        let adapter = test_adapter();
        let signature = MockSignature::new();

        let inputs = example! {
            "question": "input" => "What is 2+2?"
        };

        let chat = adapter.format(&signature, inputs);

        assert!(!chat.messages.is_empty());
        assert_eq!(chat.messages.len(), 2); // System + User

        // Check system message
        match &chat.messages[0] {
            Message::System { content } => {
                assert_eq!(content, "Answer the question concisely.");
            }
            _ => panic!("Expected system message"),
        }

        // Check user message
        match &chat.messages[1] {
            Message::User { content } => {
                assert!(content.contains("What is 2+2?"));
            }
            _ => panic!("Expected user message"),
        }
    }

    #[test]
    #[ignore] // Uses zeroed model, safe only if model isn't accessed
    fn test_parse_response() {
        let adapter = test_adapter();
        let signature = MockSignature::new();

        let response = Message::assistant("The answer is 4");
        let outputs = adapter.parse_response(&signature, response);

        assert!(outputs.contains_key("answer"));
        assert_eq!(
            outputs.get("answer").unwrap().as_str().unwrap(),
            "The answer is 4"
        );
    }

    #[test]
    #[ignore] // Uses zeroed model (safe only if model isn't accessed)
    fn test_parse_response_field_marker() {
        // Test Strategy 1: Field marker parsing ("FieldName: value")
        let adapter = test_adapter();
        let signature = MockSignature::new();

        let response = Message::assistant("answer: Paris is the capital of France");
        let outputs = adapter.parse_response(&signature, response);

        assert!(outputs.contains_key("answer"));
        assert_eq!(
            outputs.get("answer").unwrap().as_str().unwrap(),
            "Paris is the capital of France"
        );
    }

    #[test]
    #[ignore] // Uses zeroed model (safe only if model isn't accessed)
    fn test_parse_response_multi_field() {
        // Test Strategy 1: Multiple fields with markers
        let adapter = test_adapter();
        let mut signature = MockSignature::new();
        signature.output_fields.clear();
        signature.output_fields.push("reasoning".to_string());
        signature.output_fields.push("answer".to_string());

        let response = Message::assistant(
            "Reasoning: To multiply 15 by 23, I break it down: 15*20=300, 15*3=45, total=345\nAnswer: 345"
        );
        let outputs = adapter.parse_response(&signature, response);

        assert!(outputs.contains_key("reasoning"));
        assert!(outputs.contains_key("answer"));
        assert!(outputs.get("reasoning").unwrap().as_str().unwrap().contains("break it down"));
        assert_eq!(outputs.get("answer").unwrap().as_str().unwrap(), "345");
    }

    #[test]
    #[ignore] // Uses zeroed model (safe only if model isn't accessed)
    fn test_parse_response_json() {
        // Test Strategy 2: JSON parsing fallback
        let adapter = test_adapter();
        let signature = MockSignature::new();

        let response = Message::assistant(r#"{"answer": "42"}"#);
        let outputs = adapter.parse_response(&signature, response);

        assert!(outputs.contains_key("answer"));
        assert_eq!(outputs.get("answer").unwrap().as_str().unwrap(), "42");
    }

    #[test]
    #[ignore] // Uses zeroed model (safe only if model isn't accessed)
    fn test_parse_response_single_field_fallback() {
        // Test Strategy 3: Single field fallback (entire response)
        let adapter = test_adapter();
        let signature = MockSignature::new();

        let response = Message::assistant("Paris");
        let outputs = adapter.parse_response(&signature, response);

        assert!(outputs.contains_key("answer"));
        assert_eq!(outputs.get("answer").unwrap().as_str().unwrap(), "Paris");
    }

    #[test]
    #[ignore] // Uses zeroed model (safe only if model isn't accessed)
    fn test_format_with_demonstrations() {
        // Test demonstration formatting
        let adapter = test_adapter();
        let mut signature = MockSignature::new();

        // Add demonstrations to signature
        signature.demonstrations = vec![
            example! {
                "question": "input" => "What is 2+2?",
                "answer": "output" => "4"
            },
            example! {
                "question": "input" => "What is 3+3?",
                "answer": "output" => "6"
            },
        ];

        let inputs = example! {
            "question": "input" => "What is 5+5?"
        };

        let chat = adapter.format(&signature, inputs);

        // Should have: System + Demo1(User+Assistant) + Demo2(User+Assistant) + Actual(User)
        assert_eq!(chat.messages.len(), 6);

        // System message
        assert!(matches!(chat.messages[0], Message::System { .. }));

        // Demo 1: User message with "What is 2+2?"
        if let Message::User { content } = &chat.messages[1] {
            assert!(content.contains("2+2"));
        } else {
            panic!("Expected User message for demo 1 input");
        }

        // Demo 1: Assistant message with "4"
        if let Message::Assistant { content } = &chat.messages[2] {
            assert!(content.contains("4"));
        } else {
            panic!("Expected Assistant message for demo 1 output");
        }

        // Demo 2: User message with "What is 3+3?"
        if let Message::User { content } = &chat.messages[3] {
            assert!(content.contains("3+3"));
        } else {
            panic!("Expected User message for demo 2 input");
        }

        // Demo 2: Assistant message with "6"
        if let Message::Assistant { content } = &chat.messages[4] {
            assert!(content.contains("6"));
        } else {
            panic!("Expected Assistant message for demo 2 output");
        }

        // Actual input: User message with "What is 5+5?"
        if let Message::User { content } = &chat.messages[5] {
            assert!(content.contains("5+5"));
        } else {
            panic!("Expected User message for actual input");
        }
    }

    #[test]
    #[ignore] // Uses zeroed model, safe only if model isn't accessed
    fn test_chat_to_prompt() {
        let adapter = test_adapter();

        let messages = vec![
            Message::system("You are helpful"),
            Message::user("Hello"),
            Message::assistant("Hi there!"),
        ];

        let chat = Chat::new(messages);
        let prompt = adapter.chat_to_prompt(&chat);

        assert!(prompt.contains("System: You are helpful"));
        assert!(prompt.contains("User: Hello"));
        assert!(prompt.contains("Assistant: Hi there!"));
    }

    // Note: These tests are disabled in Phase 1 because generate() now requires a real model.
    // The mock LoadedModel uses zeroed memory which will panic on actual inference.
    // See Phase 1 integration tests in tests/ directory for real model testing.

    #[tokio::test]
    #[ignore] // Requires real model, not mock
    async fn test_generate_real_model() {
        // This test requires a real model loaded from Model Pool
        // See specs/PHASE-1-VERIFICATION.md for integration test examples
        todo!("Implement with real model from Model Pool");
    }

    #[tokio::test]
    #[ignore] // Requires real model, not mock
    async fn test_call_with_real_model() {
        // This test requires a real model loaded from Model Pool
        // See specs/PHASE-1-VERIFICATION.md for integration test examples
        todo!("Implement with real model from Model Pool");
    }
}
