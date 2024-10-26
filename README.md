
# Linguify
## Tailored Grammar Feedback for Non-Native English Writers

Linguify leverages a T5 model and OpenAI's GPT models to correct grammar in sentences and provide expert-level feedback. The system uses the JFLEG dataset for evaluation, offering targeted grammar corrections and detailed feedback for non-native English writers.

## Features
- **Grammar Correction**: Uses a fine-tuned T5 model (`t5-base` by default) for precise grammar correction.
- **Expert Feedback**: Generates expert-level grammatical feedback with GPT-4 through the OpenAI API.
- **Dataset Support**: Efficiently handles the JFLEG dataset for training, validation, and testing through Hugging Face's `datasets` library.

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- [OpenAI API Key](https://platform.openai.com/account/api-keys) (required for feedback generation)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/schandupatla4/Linguify.git
   cd Linguify
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the OpenAI API Key**:
   - Create a `.env` file in the project root and add your OpenAI API key:
     ```plaintext
     OPENAI_API_KEY=your-api-key
     ```

4. **Run the Application**:
   To run the entire pipeline (loading data, training, correcting grammar, evaluating, and providing feedback), use:
   ```bash
   python linguify_project.py
   ```

## Project Structure

```
Linguify/
├── notebook/
│   └── Linguify_Project.ipynb             # Configuration file with model settings and API key management
├── .env                      # Environment variables (store OpenAI API key here)
├── README.md                 # Project documentation
└── .gitignore                # Files and directories to ignore in Git
```

## Usage

1. **Grammar Correction**: Automatically corrects grammar for sentences in the JFLEG dataset using the T5 model.
2. **GLEU Evaluation**: Computes GLEU score to evaluate correction quality.
3. **Feedback Generation**: Provides feedback on corrected sentences, limited to a configurable number of samples.

Example usage steps are incorporated into `main.py`, which orchestrates the entire workflow.

