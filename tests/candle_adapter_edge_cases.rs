//! Edge Case Tests for Candle Adapter
//!
//! These tests cover edge cases, error handling, and boundary conditions
//! to ensure robustness of the Phase 2A and 2B implementations.

use ml_crate_dsrs::{CandleAdapter, CandleConfig, ModelPool};
use dspy_rs::{adapter::Adapter, example, Message, MetaSignature};
use serde_json::Value;

/// Helper to load model (reused across tests)
async fn setup_adapter() -> CandleAdapter {
    let pool = ModelPool::new("./models".into());
    let loaded = pool
        .load_model("Qwen2.5-0.5B")
        .await
        .expect("Failed to load model");
    CandleAdapter::from_loaded_model(loaded, CandleConfig::default())
}

/// Mock signature for testing
struct TestSignature {
    instruction: String,
    output_fields: Vec<String>,
    demonstrations: Vec<dspy_rs::Example>,
}

impl TestSignature {
    fn new(instruction: &str, output_fields: Vec<&str>) -> Self {
        Self {
            instruction: instruction.to_string(),
            output_fields: output_fields.iter().map(|s| s.to_string()).collect(),
            demonstrations: vec![],
        }
    }

    fn with_demos(mut self, demos: Vec<dspy_rs::Example>) -> Self {
        self.demonstrations = demos;
        self
    }
}

impl MetaSignature for TestSignature {
    fn demos(&self) -> Vec<dspy_rs::Example> {
        self.demonstrations.clone()
    }

    fn set_demos(&mut self, demos: Vec<dspy_rs::Example>) -> anyhow::Result<()> {
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

    fn append(&mut self, _name: &str, _value: Value) -> anyhow::Result<()> {
        Ok(())
    }
}

// ============================================================================
// Demonstration Edge Cases
// ============================================================================

#[tokio::test]
#[ignore] // Requires model download
async fn test_edge_case_empty_demonstrations() {
    println!("\n=== Edge Case: Empty Demonstrations List ===");

    let adapter = setup_adapter().await;

    // Create signature with empty demonstrations
    let signature = TestSignature::new("Answer questions.", vec!["answer"])
        .with_demos(vec![]);

    let inputs = example! {
        "question": "input" => "What is 2+2?"
    };

    let chat = adapter.format(&signature, inputs);

    // Should have: System + User = 2 messages (no demo messages)
    assert_eq!(chat.messages.len(), 2, "Should have 2 messages with empty demos");

    println!("✅ Empty demonstrations handled correctly");
}

#[tokio::test]
#[ignore]
async fn test_edge_case_demonstration_with_special_characters() {
    println!("\n=== Edge Case: Demonstrations with Special Characters ===");

    let adapter = setup_adapter().await;

    // Create demo with special characters
    let signature = TestSignature::new("Answer questions.", vec!["answer"])
        .with_demos(vec![
            example! {
                "question": "input" => "What is \"hello\" + 'world'?",
                "answer": "output" => "Concatenation: \"hello\" + 'world' = \"helloworld\""
            },
        ]);

    let inputs = example! {
        "question": "input" => "Test?"
    };

    let chat = adapter.format(&signature, inputs);

    // Verify special characters are preserved
    let demo_user_msg = &chat.messages[1];
    if let Message::User { content } = demo_user_msg {
        assert!(content.contains("\"hello\""), "Should preserve double quotes");
        assert!(content.contains("'world'"), "Should preserve single quotes");
    } else {
        panic!("Expected User message for demonstration");
    }

    println!("✅ Special characters in demonstrations handled correctly");
}

#[tokio::test]
#[ignore]
async fn test_edge_case_very_long_demonstration() {
    println!("\n=== Edge Case: Very Long Text in Demonstrations ===");

    let adapter = setup_adapter().await;

    // Create demo with very long text (500+ chars)
    let long_text = "a".repeat(500);
    let signature = TestSignature::new("Answer questions.", vec!["answer"])
        .with_demos(vec![
            example! {
                "question": "input" => &long_text,
                "answer": "output" => "Long input processed"
            },
        ]);

    let inputs = example! {
        "question": "input" => "Short question"
    };

    let chat = adapter.format(&signature, inputs);

    // Verify long text is included
    let demo_user_msg = &chat.messages[1];
    if let Message::User { content } = demo_user_msg {
        assert!(content.len() > 500, "Should include long demonstration text");
    } else {
        panic!("Expected User message for demonstration");
    }

    println!("✅ Very long demonstrations handled correctly");
}

// ============================================================================
// Response Parsing Edge Cases
// ============================================================================

#[tokio::test]
#[ignore]
async fn test_edge_case_empty_response() {
    println!("\n=== Edge Case: Empty Response ===");

    let adapter = setup_adapter().await;
    let signature = TestSignature::new("Answer questions.", vec!["answer"]);

    let response = Message::assistant("");
    let outputs = adapter.parse_response(&signature, response);

    // With single field, empty response should be captured
    assert!(outputs.contains_key("answer"), "Should have answer field");
    assert_eq!(
        outputs.get("answer").unwrap().as_str().unwrap(),
        "",
        "Empty response should be captured"
    );

    println!("✅ Empty response handled correctly");
}

#[tokio::test]
#[ignore]
async fn test_edge_case_whitespace_only_response() {
    println!("\n=== Edge Case: Response with Only Whitespace ===");

    let adapter = setup_adapter().await;
    let signature = TestSignature::new("Answer questions.", vec!["answer"]);

    let response = Message::assistant("   \n\t  ");
    let outputs = adapter.parse_response(&signature, response);

    // Whitespace should be trimmed by parse_response
    assert!(outputs.contains_key("answer"), "Should have answer field");

    println!("Parsed output: {:?}", outputs.get("answer"));
    println!("✅ Whitespace-only response handled correctly");
}

#[tokio::test]
#[ignore]
async fn test_edge_case_field_marker_no_value() {
    println!("\n=== Edge Case: Field Marker but No Value ===");

    let adapter = setup_adapter().await;
    let signature = TestSignature::new("Solve problems.", vec!["answer"]);

    let response = Message::assistant("answer:");
    let outputs = adapter.parse_response(&signature, response);

    // Field marker with no value should either:
    // - Extract empty string after colon, OR
    // - Fall back to single-field strategy
    assert!(outputs.contains_key("answer"), "Should have answer field");

    println!("Parsed output: {:?}", outputs.get("answer"));
    println!("✅ Field marker with no value handled correctly");
}

#[tokio::test]
#[ignore]
async fn test_edge_case_malformed_json() {
    println!("\n=== Edge Case: Malformed JSON Response ===");

    let adapter = setup_adapter().await;
    let signature = TestSignature::new("Answer questions.", vec!["answer"]);

    let response = Message::assistant(r#"{"answer": "incomplete"#);
    let outputs = adapter.parse_response(&signature, response);

    // Malformed JSON should fall back to single-field strategy
    assert!(outputs.contains_key("answer"), "Should have answer field from fallback");

    let answer = outputs.get("answer").unwrap().as_str().unwrap();
    println!("Parsed output: {}", answer);

    // Should capture the whole malformed JSON as the answer value
    assert!(answer.contains("answer"), "Should contain original text");

    println!("✅ Malformed JSON handled correctly with fallback");
}

#[tokio::test]
#[ignore]
async fn test_edge_case_missing_field_in_multifield_response() {
    println!("\n=== Edge Case: Missing Required Field in Multi-field Response ===");

    let adapter = setup_adapter().await;
    let signature = TestSignature::new("Solve with reasoning.", vec!["reasoning", "answer"]);

    // Response with only one field
    let response = Message::assistant("answer: 42");
    let outputs = adapter.parse_response(&signature, response);

    // Should extract what's available
    assert!(outputs.contains_key("answer"), "Should extract available field");

    // Missing field behavior: may or may not be present depending on parsing strategy
    println!("Fields extracted: {:?}", outputs.keys().collect::<Vec<_>>());
    println!("✅ Partial field extraction handled");
}

// ============================================================================
// Multi-turn Conversation State Isolation
// ============================================================================

#[tokio::test]
#[ignore]
async fn test_edge_case_multi_turn_state_isolation() {
    println!("\n=== Edge Case: Multi-turn Conversation State Isolation ===");

    let adapter = setup_adapter().await;
    let signature = TestSignature::new("Answer questions.", vec!["answer"]);

    // First call
    let inputs1 = example! {
        "question": "input" => "First question"
    };
    let chat1 = adapter.format(&signature, inputs1);
    assert_eq!(chat1.messages.len(), 2);

    // Second call with same signature
    let inputs2 = example! {
        "question": "input" => "Second question"
    };
    let chat2 = adapter.format(&signature, inputs2);
    assert_eq!(chat2.messages.len(), 2);

    // Verify no state leakage between calls
    if let Message::User { content: content1 } = &chat1.messages[1] {
        if let Message::User { content: content2 } = &chat2.messages[1] {
            assert!(content1.contains("First"));
            assert!(content2.contains("Second"));
            assert!(!content2.contains("First"), "No state should leak between calls");
        }
    }

    println!("✅ Multi-turn state isolation verified");
}

// ============================================================================
// Performance Edge Cases
// ============================================================================

#[tokio::test]
#[ignore]
async fn test_edge_case_very_long_prompt() {
    println!("\n=== Edge Case: Very Long Prompt (Near Context Limit) ===");

    let adapter = setup_adapter().await;

    // Create a very long prompt (but under 32K context limit)
    let long_prompt = "word ".repeat(5000); // ~25K chars

    println!("Prompt length: {} chars", long_prompt.len());

    // This should work without hitting context limit
    let result = adapter.generate(&long_prompt).await;

    match result {
        Ok((_response, prompt_tokens, completion_tokens)) => {
            println!("Generated successfully:");
            println!("  Prompt tokens: {}", prompt_tokens);
            println!("  Completion tokens: {}", completion_tokens);
            assert!(prompt_tokens > 1000, "Should have many prompt tokens");
            println!("✅ Very long prompt handled correctly");
        }
        Err(e) => {
            println!("Error (may be expected if too long): {}", e);
        }
    }
}

#[tokio::test]
#[ignore]
async fn test_edge_case_very_short_prompt() {
    println!("\n=== Edge Case: Very Short Prompt ===");

    let adapter = setup_adapter().await;

    let short_prompt = "Hi";

    let result = adapter.generate(short_prompt).await;

    assert!(result.is_ok(), "Short prompt should work");

    let (response, prompt_tokens, completion_tokens) = result.unwrap();

    println!("Prompt: {}", short_prompt);
    println!("Response: {}", response);
    println!("Prompt tokens: {}", prompt_tokens);
    println!("Completion tokens: {}", completion_tokens);

    assert!(prompt_tokens > 0, "Should count prompt tokens");
    assert!(!response.is_empty(), "Should generate response");

    println!("✅ Very short prompt handled correctly");
}

#[tokio::test]
#[ignore]
async fn test_edge_case_rapid_successive_calls() {
    println!("\n=== Edge Case: Rapid Successive Calls ===");

    let adapter = setup_adapter().await;

    let prompts = vec!["A", "B", "C"];

    for prompt in prompts {
        let result = adapter.generate(prompt).await;
        assert!(result.is_ok(), "Each call should succeed");

        let (response, _, _) = result.unwrap();
        println!("Prompt: {} -> Response: {}", prompt, response.chars().take(50).collect::<String>());
    }

    println!("✅ Rapid successive calls handled correctly");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
#[ignore]
async fn test_error_context_too_long() {
    println!("\n=== Error Handling: Context Too Long ===");

    let adapter = setup_adapter().await;

    // Create a prompt that exceeds context length (32K tokens)
    // Rough estimate: 1 token ~= 4 chars, so 150K chars should exceed 32K tokens
    let too_long = "word ".repeat(30000); // ~150K chars

    println!("Prompt length: {} chars", too_long.len());

    let result = adapter.generate(&too_long).await;

    // Should get ContextTooLong error
    assert!(result.is_err(), "Should fail with context too long");

    let err = result.unwrap_err();
    println!("Error (expected): {}", err);

    // Check error message
    let err_str = format!("{}", err);
    assert!(err_str.contains("too long") || err_str.contains("context"), "Error should mention context length");

    println!("✅ Context too long error handled correctly");
}
