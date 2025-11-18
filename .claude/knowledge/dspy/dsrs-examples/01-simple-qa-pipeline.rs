// Example: Simple QA Pipeline with Rating
// Demonstrates: Signatures, Modules, Module composition, Chain-of-Thought

use anyhow::Result;
use bon::Builder;
use dspy_rs::{
    ChatAdapter, Example, LM, Module, Predict, Prediction, Predictor, Signature, configure,
    example, prediction,
};

#[Signature(cot)]  // Chain-of-thought enabled
struct QASignature {
    #[input]
    pub question: String,

    #[output]
    pub answer: String,
}

#[Signature]
struct RateSignature {
    /// Rate the answer on a scale of 1(very bad) to 10(very good)

    #[input]
    pub question: String,

    #[input]
    pub answer: String,

    #[output]
    pub rating: i8,
}

// Composite module: QA + Rating
#[derive(Builder)]
pub struct QARater {
    #[builder(default = Predict::new(QASignature::new()))]
    pub answerer: Predict,
    #[builder(default = Predict::new(RateSignature::new()))]
    pub rater: Predict,
}

impl Module for QARater {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        // Step 1: Generate answer
        let answerer_prediction = self.answerer.forward(inputs.clone()).await?;

        let question = inputs.data.get("question").unwrap().clone();
        let answer = answerer_prediction.data.get("answer").unwrap().clone();

        // Step 2: Rate the answer
        let inputs = example! {
            "question": "input" => question.clone(),
            "answer": "output" => answer.clone()
        };

        let rating_prediction = self.rater.forward(inputs).await?;

        // Combine results
        Ok(prediction! {
            "answer"=> answer,
            "question"=> question,
            "rating"=> rating_prediction.data.get("rating").unwrap().clone(),
        }
        .set_lm_usage(rating_prediction.lm_usage))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Configure LM
    configure(
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build()
            .await
            .unwrap(),
        ChatAdapter,
    );

    // Create input
    let example = example! {
        "question": "input" => "What is the capital of France?",
    };

    // Run pipeline
    let qa_rater = QARater::builder().build();
    let prediction = qa_rater.forward(example).await.unwrap();
    println!("{prediction:?}");

    Ok(())
}

/* Key Patterns:
 * 1. Multiple signatures for different tasks
 * 2. Chain-of-thought via #[Signature(cot)]
 * 3. Module composition (answerer + rater)
 * 4. Async forward() implementation
 * 5. Builder pattern for module construction
 */
