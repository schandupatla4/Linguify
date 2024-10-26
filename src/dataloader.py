from datasets import load_dataset, DatasetDict
from transformers import T5Tokenizer

def load_and_preprocess_data():
    dataset = load_dataset("jhu-clsp/jfleg")
    train_test_split = dataset['validation'].train_test_split(test_size=0.2)
    dataset = DatasetDict({
        'train': train_test_split['train'],
        'validation': train_test_split['test'],
        'test': dataset['test']
    })
    return dataset, dataset['test'][:10]  # Example subset
