from src.data_loader import load_and_preprocess_data
from src.model import train_model, correct_grammar
from src.evaluation import evaluate_gleu
from src.feedback import provide_feedback
from config.config import OPENAI_API_KEY, NUM_SENTENCES_FOR_FEEDBACK
import openai

def main():
    # Set OpenAI API Key
    openai.api_key = OPENAI_API_KEY

    # Step 1: Load and preprocess data
    print("Loading and preprocessing data...")
    dataset, test_subset = load_and_preprocess_data()

    # Step 2: Train the model
    print("Training the model...")
    model, tokenizer = train_model(dataset)

    # Step 3: Correct grammar for a subset of test data
    print("Correcting grammar for test subset...")
    corrected_sentences = correct_grammar(model, tokenizer, test_subset)

    # Step 4: Evaluate the corrected sentences with GLEU
    print("Evaluating GLEU score on corrected sentences...")
    avg_gleu_score = evaluate_gleu(corrected_sentences, test_subset)
    print(f"Average GLEU Score: {avg_gleu_score:.4f}")

    # Step 5: Generate feedback on corrected sentences
    print(f"Generating feedback on the first {NUM_SENTENCES_FOR_FEEDBACK} corrected sentences...")
    feedback_list = provide_feedback(corrected_sentences, NUM_SENTENCES_FOR_FEEDBACK)

    # Optional: Print feedback summary
    print("\nFeedback Summary:")
    for i, (sentence, feedback) in enumerate(feedback_list):
        print(f"Feedback {i+1} for '{sentence}':\n{feedback}\n{'='*80}")

if __name__ == "__main__":
    main()
