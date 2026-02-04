import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model

from src.models.load_model import load_base_model
from src.lora.lora_config import get_lora_config


def preprocess(example, tokenizer):
    prompt = f"Summarize:\n{example['article']}\nSummary:"
    target = example["highlights"]

    text = prompt + target
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512
    )


def train_lora():
    model, tokenizer = load_base_model()
    lora_config = get_lora_config(rank=8)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("cnn_dailymail", "3.0.0")
    dataset = dataset["train"].shuffle(seed=42).select(range(200))

    dataset = dataset.map(
        lambda x: preprocess(x, tokenizer),
        remove_columns=dataset.column_names
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir="experiments/lora_summarization",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()

    model.save_pretrained("experiments/lora_summarization")


if __name__ == "__main__":
    train_lora()

