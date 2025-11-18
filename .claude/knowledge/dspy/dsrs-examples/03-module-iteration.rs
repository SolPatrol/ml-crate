// Example: Module Parameter Iteration and Updates
// Demonstrates: Nested modules, parameter iteration, dynamic updates, Optimizable

use anyhow::Result;
use bon::Builder;
use dspy_rs::{
    Example, Module, Optimizable, Predict, Prediction, Predictor, Signature, hashmap, prediction,
};

#[Signature(cot)]
struct QASignature {
    #[input]
    pub question: String,

    #[output]
    pub answer: String,
}

#[Signature]
struct RateSignature {
    #[input]
    pub question: String,

    #[input]
    pub answer: String,

    #[output]
    pub rating: i8,
}

// Composite module with optimizable parameters
#[derive(Builder, Optimizable)]
pub struct QARater {
    #[parameter]
    #[builder(default = Predict::new(QASignature::new()))]
    pub answerer: Predict,

    #[parameter]
    #[builder(default = Predict::new(RateSignature::new()))]
    pub rater: Predict,
}

// Nested module structure
#[derive(Builder, Optimizable)]
pub struct NestedModule {
    #[parameter]
    #[builder(default = QARater::builder().build())]
    pub qa_outer: QARater,

    #[parameter]
    #[builder(default = QARater::builder().build())]
    pub qa_inner: QARater,

    #[parameter]
    #[builder(default = Predict::new(QASignature::new()))]
    pub extra: Predict,
}

impl Module for QARater {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        let answerer_prediction = self.answerer.forward(inputs.clone()).await?;

        let question = inputs.data.get("question").unwrap().clone();
        let answer = answerer_prediction.data.get("answer").unwrap().clone();

        let inputs = Example::new(
            hashmap! {
                "answer".to_string() => answer.clone(),
                "question".to_string() => question.clone()
            },
            vec!["answer".to_string(), "question".to_string()],
            vec![],
        );
        let rating_prediction = self.rater.forward(inputs).await?;
        Ok(prediction! {
            "answer"=> answer,
            "question"=> question,
            "rating"=> rating_prediction.data.get("rating").unwrap().clone(),
        }
        .set_lm_usage(rating_prediction.lm_usage))
    }
}

#[tokio::main]
async fn main() {
    // Example 1: Iterate and update flat module
    let mut qa_rater = QARater::builder().build();
    for (name, param) in qa_rater.parameters() {
        param
            .update_signature_instruction("Updated instruction for ".to_string() + &name)
            .unwrap();
    }
    println!(
        "single.answerer -> {}",
        qa_rater.answerer.signature.instruction()
    );
    println!(
        "single.rater    -> {}",
        qa_rater.rater.signature.instruction()
    );

    // Example 2: Iterate and update nested module structure
    let mut nested = NestedModule::builder().build();
    for (name, param) in nested.parameters() {
        param
            .update_signature_instruction("Deep updated: ".to_string() + &name)
            .unwrap();
    }

    println!(
        "nested.qa_outer.answerer -> {}",
        nested.qa_outer.answerer.signature.instruction()
    );
    println!(
        "nested.qa_outer.rater    -> {}",
        nested.qa_outer.rater.signature.instruction()
    );
    println!(
        "nested.qa_inner.answerer -> {}",
        nested.qa_inner.answerer.signature.instruction()
    );
    println!(
        "nested.qa_inner.rater    -> {}",
        nested.qa_inner.rater.signature.instruction()
    );
    println!(
        "nested.extra    -> {}",
        nested.extra.signature.instruction()
    );
}

/* Key Patterns:
 * 1. #[derive(Optimizable)] enables parameter iteration
 * 2. #[parameter] marks components for optimization
 * 3. parameters() method provides iterator
 * 4. update_signature_instruction() modifies prompts
 * 5. Works with both flat and deeply nested structures
 * 6. Useful for custom optimization strategies
 */
