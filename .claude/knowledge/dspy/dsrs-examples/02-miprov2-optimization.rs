// Example: MIPROv2 Optimizer
// Demonstrates: Optimization, Evaluation, Data loading, Optimizable trait

use anyhow::Result;
use bon::Builder;
use dspy_rs::{
    ChatAdapter, DataLoader, Evaluator, Example, LM, MIPROv2, Module, Optimizable, Optimizer,
    Predict, Prediction, Predictor, Signature, configure, example,
};

#[Signature]
struct QuestionAnswering {
    /// Answer the question accurately and concisely.

    #[input]
    pub question: String,

    #[output]
    pub answer: String,
}

#[derive(Builder, Optimizable)]
pub struct SimpleQA {
    #[parameter]  // Marks this as optimizable
    #[builder(default = Predict::new(QuestionAnswering::new()))]
    pub answerer: Predict,
}

impl Module for SimpleQA {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        self.answerer.forward(inputs).await
    }
}

// Custom metric implementation
impl Evaluator for SimpleQA {
    async fn metric(&self, example: &Example, prediction: &Prediction) -> f32 {
        let expected = example
            .data
            .get("answer")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let predicted = prediction
            .data
            .get("answer")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let expected_normalized = expected.to_lowercase().trim().to_string();
        let predicted_normalized = predicted.to_lowercase().trim().to_string();

        if expected_normalized == predicted_normalized {
            1.0
        } else {
            if expected_normalized.contains(&predicted_normalized)
                || predicted_normalized.contains(&expected_normalized)
            {
                0.5
            } else {
                0.0
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== MIPROv2 Optimizer Example ===\n");

    configure(LM::default(), ChatAdapter);

    // Load data from HuggingFace
    println!("Loading training data from HuggingFace...");
    let train_examples = DataLoader::load_hf(
        "hotpotqa/hotpot_qa",
        vec!["question".to_string()],
        vec!["answer".to_string()],
        "fullwiki",
        "validation",
        true,
    )?;

    let train_subset = train_examples[..15].to_vec();
    println!("Using {} training examples\n", train_subset.len());

    let mut qa_module = SimpleQA::builder().build();

    println!("Initial instruction:");
    println!(
        "  \"{}\"\n",
        qa_module.answerer.get_signature().instruction()
    );

    // Baseline evaluation
    println!("Evaluating baseline performance...");
    let baseline_score = qa_module.evaluate(train_subset[..5].to_vec()).await;
    println!("Baseline score: {:.3}\n", baseline_score);

    // Configure MIPROv2 optimizer
    let optimizer = MIPROv2::builder()
        .num_candidates(8)         // Number of prompt candidates
        .num_trials(15)            // Optimization trials
        .minibatch_size(10)        // Examples per trial
        .temperature(1.0)          // LLM temperature
        .track_stats(true)         // Track optimization stats
        .build();

    println!("Starting MIPROv2 optimization...");
    println!("This will:");
    println!("  1. Generate execution traces");
    println!("  2. Create a program description using LLM");
    println!("  3. Generate {} candidate prompts with best practices", 8);
    println!("  4. Evaluate each candidate");
    println!("  5. Select and apply the best prompt\n");

    // Run optimization
    optimizer
        .compile(&mut qa_module, train_subset.clone())
        .await?;

    println!("\nOptimized instruction:");
    println!(
        "  \"{}\"\n",
        qa_module.answerer.get_signature().instruction()
    );

    // Evaluate optimized version
    println!("Evaluating optimized performance...");
    let optimized_score = qa_module.evaluate(train_subset[..5].to_vec()).await;
    println!("Optimized score: {:.3}", optimized_score);

    let improvement = ((optimized_score - baseline_score) / baseline_score) * 100.0;
    println!(
        "\n✓ Improvement: {:.1}% ({:.3} -> {:.3})",
        improvement, baseline_score, optimized_score
    );

    // Test on new example
    println!("\n--- Testing on a new example ---");
    let test_example = example! {
        "question": "input" => "What is the capital of France?",
    };

    let result = qa_module.forward(test_example).await?;
    println!("Question: What is the capital of France?");
    println!("Answer: {}", result.get("answer", None));

    println!("\n=== Example Complete ===");
    Ok(())
}

/* Key Patterns:
 * 1. #[derive(Optimizable)] enables optimization
 * 2. #[parameter] marks optimizable components
 * 3. Custom Evaluator trait for metrics
 * 4. DataLoader for HuggingFace datasets
 * 5. MIPROv2 optimizer configuration
 * 6. Baseline vs optimized comparison
 */
