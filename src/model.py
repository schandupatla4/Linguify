from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments

def train_model(dataset):
    model_name = "t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_name).to("cuda")
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir="./t5_finetuned_jfleg",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )
    trainer.train()
    
    return model, tokenizer
