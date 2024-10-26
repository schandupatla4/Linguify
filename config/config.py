import os
from transformers import TrainingArguments

# Paths
OUTPUT_DIR = "./t5_finetuned_jfleg"  # Directory to save model checkpoints
MODEL_NAME = "t5-base"  # Model name for loading the pretrained model

# OpenAI API Key
# Store the key in an environment variable and load it here
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-default-key-here")

# Training Arguments
TRAINING_ARGS = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    report_to="none"
)

# Evaluation Settings
NUM_SENTENCES_FOR_FEEDBACK = 10  # Number of sentences to provide feedback on
