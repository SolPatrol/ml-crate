// Debug test to investigate gibberish output
// Run with: cargo test --test debug_inference -- --nocapture --ignored

use candle_core::IndexOp;
use ml_crate_dsrs::{CandleConfig, ModelPool};

#[tokio::test]
#[ignore]
async fn debug_tokenization() {
    let pool = ModelPool::new("./models".into());
    let loaded = pool.load_model("Qwen2.5-0.5B").await.unwrap();

    let test_text = "What is 2+2?";

    // Encode
    let encoding = loaded.tokenizer.encode(test_text, true).unwrap();
    let tokens = encoding.get_ids();

    println!("Input text: {}", test_text);
    println!("Input tokens: {:?}", tokens);

    // Decode
    let decoded = loaded.tokenizer.decode(tokens, true).unwrap();
    println!("Decoded: {}", decoded);

    // Try decoding individual tokens to see what token ID maps to "l"
    for &token_id in tokens {
        let single = loaded.tokenizer.decode(&[token_id], false).unwrap();
        println!("Token {} -> '{}'", token_id, single);
    }

    // Check what token 75 decodes to (likely "l")
    println!("\nChecking common token IDs:");
    for id in [75, 76, 77, 108, 109, 110] {
        let decoded = loaded.tokenizer.decode(&[id], false).unwrap();
        println!("Token {} -> '{}'", id, decoded);
    }
}

#[tokio::test]
#[ignore]
async fn debug_model_forward() {
    use candle_core::Tensor;

    let pool = ModelPool::new("./models".into());
    let loaded = pool.load_model("Qwen2.5-0.5B").await.unwrap();
    let _config = CandleConfig::default();

    let test_text = "Hello";

    // Tokenize
    let encoding = loaded.tokenizer.encode(test_text, true).unwrap();
    let input_tokens: Vec<u32> = encoding.get_ids().to_vec();

    println!("Input: {}", test_text);
    println!("Tokens: {:?}", input_tokens);

    // Run one forward pass
    let mut model_guard = loaded.model.lock().unwrap();
    model_guard.clear_kv_cache();

    let input_tensor = Tensor::new(&input_tokens[..], &loaded.device)
        .unwrap()
        .reshape((1, input_tokens.len()))
        .unwrap();

    println!("Input tensor shape: {:?}", input_tensor.shape());

    let logits = model_guard.forward(&input_tensor, 0).unwrap();

    println!("Logits shape: {:?}", logits.shape());

    // Get logits for last position
    let seq_len = logits.dim(1).unwrap();
    let last_logits = logits.i((.., seq_len - 1, ..)).unwrap().squeeze(0).unwrap();

    println!("Last logits shape: {:?}", last_logits.shape());

    // Convert to F32 and get top 10 tokens
    use candle_core::DType;
    let last_logits_f32 = if last_logits.dtype() == DType::F16 {
        last_logits.to_dtype(DType::F32).unwrap()
    } else {
        last_logits
    };

    let logits_vec = last_logits_f32.to_vec1::<f32>().unwrap();

    // Get top 10 tokens by logit value
    let mut indexed: Vec<(usize, f32)> = logits_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop 10 tokens by logit:");
    for (idx, &(token_id, logit_val)) in indexed.iter().take(10).enumerate() {
        let decoded = loaded.tokenizer.decode(&[token_id as u32], false).unwrap();
        println!("{}. Token {} (logit={:.2}): '{}'", idx + 1, token_id, logit_val, decoded);
    }
}
