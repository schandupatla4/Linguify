import time
import openai

def provide_feedback(sentences, num_sentences=10):
    feedback_list = []
    for i, sentence in enumerate(sentences[:num_sentences]):
        prompt = f"The sentence is: '{sentence}'\n" \
                 f"Please provide detailed feedback on the grammatical correctness of this sentence and suggest improvements if necessary."
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a grammar expert."},
                    {"role": "user", "content": prompt}
                ]
            )
            feedback = response['choices'][0]['message']['content']
            feedback_list.append((sentence, feedback))
            print(f"Feedback {i+1} for: '{sentence}'\n{feedback}")
            print("=" * 80)
        
        except Exception as e:
            print(f"Error for sentence '{sentence}': {e}")
        
        time.sleep(1)
    
    return feedback_list
