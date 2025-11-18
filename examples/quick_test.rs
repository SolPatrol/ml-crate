//! Quick test to verify Qwen2.5-0.5B model loads and generates text
//!
//! Run with: cargo run --example quick_test

use ml_crate_dsrs::{CandleAdapter, CandleConfig, ModelPool};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing for debug output
    tracing_subscriber::fmt::init();

    println!("\n=== Phase 1 Quick Test: Qwen2.5-0.5B ===\n");

    // Step 1: Load model from Model Pool
    println!("📦 Loading model from disk...");
    let pool = ModelPool::new("./models".into());
    let loaded = pool.load_model("Qwen2.5-0.5B").await?;

    println!("✅ Model loaded successfully!");
    println!("   - Model: {}", loaded.model_name);
    println!("   - Device: {:?}", loaded.device);

    // Step 2: Create adapter with reduced max_tokens for faster testing
    println!("\n🔧 Creating adapter...");
    let config = CandleConfig::default().with_max_tokens(32); // Use 32 tokens for quick testing
    let adapter = CandleAdapter::from_loaded_model(loaded, config);
    println!("✅ Adapter created!");
    println!("   - Max tokens: 32");
    println!("   - Temperature: 0.7");
    println!("   - Context length: 32768");

    // Step 3: Test generation with performance measurement
    println!("\n🤖 Testing inference...");
    let prompt = "What is 2+2?";
    println!("   Prompt: \"{}\"", prompt);

    let start = std::time::Instant::now();
    let (response, prompt_tokens, completion_tokens) =
        adapter.generate(prompt).await?;
    let duration = start.elapsed();

    let tokens_per_sec = completion_tokens as f64 / duration.as_secs_f64();

    println!("\n✅ Generation complete!");
    println!("   Response: \"{}\"", response);
    println!("   Tokens:");
    println!("     - Prompt: {}", prompt_tokens);
    println!("     - Completion: {}", completion_tokens);
    println!("     - Total: {}", prompt_tokens + completion_tokens);
    println!("   Performance:");
    println!("     - Duration: {:.2}s", duration.as_secs_f64());
    println!("     - Tokens/sec: {:.2}", tokens_per_sec);

    // Step 4: Test with longer prompt
    println!("\n🤖 Testing longer prompt...");
    let long_prompt = "Tell me a very short story about a robot.";
    println!("   Prompt: \"{}\"", long_prompt);

    let start2 = std::time::Instant::now();
    let (response2, p_tokens2, c_tokens2) =
        adapter.generate(long_prompt).await?;
    let duration2 = start2.elapsed();

    let tokens_per_sec2 = c_tokens2 as f64 / duration2.as_secs_f64();

    println!("\n✅ Second generation complete!");
    println!("   Response: \"{}\"", response2);
    println!("   Tokens:");
    println!("     - Prompt: {}", p_tokens2);
    println!("     - Completion: {}", c_tokens2);
    println!("     - Total: {}", p_tokens2 + c_tokens2);
    println!("   Performance:");
    println!("     - Duration: {:.2}s", duration2.as_secs_f64());
    println!("     - Tokens/sec: {:.2}", tokens_per_sec2);

    println!("\n🎉 Phase 1 Quick Test PASSED!\n");

    Ok(())
}
