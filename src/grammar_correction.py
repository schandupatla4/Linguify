import openai
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset

# Set OpenAI API key for expert LLM usage
openai.api_key = 'insert-your-api-key'  # Replace with your actual OpenAI API key

# Load T5 model for grammar correction
model_name = "t5-large"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Function to correct grammar using the T5 model
def correct_grammar(text):
    # Preparing the input for T5 model
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

    # Generation Settings
    outputs = model.generate(
        inputs.input_ids,
        max_length=512,
        num_beams=5,            
        early_stopping=True,    
        no_repeat_ngram_size=2, 
        repetition_penalty=2.0, 
        temperature=0.8,        
        top_k=50,               
        top_p=0.9               
    )

    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return corrected_text

# Function to provide expert-level feedback using OpenAI ChatCompletion API for the original sentence
def provide_feedback(original_text):
    # Use GPT to generate detailed feedback and explanations about the original sentence
    prompt = f"The sentence is: '{original_text}'\n" \
             f"Please provide detailed feedback on the grammatical correctness of this sentence and suggest improvements if necessary."

    # Using the OpenAI ChatCompletion API for GPT-4 or GPT-3.5-turbo
    response = openai.ChatCompletion.create(
        model="gpt-4",  # You can change this to gpt-3.5-turbo if needed
        messages=[
            {"role": "system", "content": "You are a grammar expert."},
            {"role": "user", "content": prompt}
        ]
    )

    feedback = response['choices'][0]['message']['content']
    return feedback


# Function to process multiple sentences from the dataset
def process_dataset(dataset, num_sentences=10):
    dataset = dataset.select(range(num_sentences)) 

    for example in dataset:
        sentence = example['sentence']  # Original sentence from the dataset
        feedback = provide_feedback(sentence)  # Get feedback from GPT

        # Output results
        print(f"Original: {sentence}")
        print(f"Feedback: {feedback}")
        print("="*80)


# Load the JFLEG dataset from Hugging Face
dataset = load_dataset("jhu-clsp/jfleg", split="test")

process_dataset(dataset, num_sentences=10)