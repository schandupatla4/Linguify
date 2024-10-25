# Linguify
# Linguify: Tailored Grammar Feedback for Non-Native English Writers

This project utilizes a T5 model and OpenAI's GPT models to correct grammar in sentences and provide expert-level feedback. The system uses the JFLEG dataset for evaluation.

## Features
- Grammar correction using a pre-trained T5 model (`t5-large`).
- Expert-level grammatical feedback using GPT-4 via the OpenAI API.
- Dataset handling for the JFLEG dataset through Hugging Face's `datasets` library.

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- [OpenAI API Key](https://platform.openai.com/account/api-keys) (required for feedback functionality)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/schandupatla4/NLPProject.git
   cd NLPProject
   ```
2. Installing Dependencies
   ```
   pip install -r requirements.txt
   ```
3.Set your OpenAI API key in src/grammar_correction.py
   ```
   openai.api_key = 'your-api-key'  # Replace with your actual API key
   ```
4.Usage(To Run)
   ```
   python src/grammar_correction.py
   ```
