import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model

from src.models.load_model import load_base_model
from src.lora.lora_config import get_lora_config

LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

def preprocess(example, tokenizer):
    prompt = f"""Text:
{example['text']}
Label:
"""
    label = LABEL_MAP[example["label"]]
    text = prompt + label

    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256
    )


def train_classification_lora():
    model, tokenizer = load_base_model()
    lora_config = get_lora_config(rank=8)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("ag_news")
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
        output_dir="experiments/lora_classification",
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
    model.save_pretrained("experiments/lora_classification")


if __name__ == "__main__":
    train_classification_lora()

