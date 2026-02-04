import torch
from peft import PeftModel
from transformers import AutoTokenizer

from src.models.load_model import load_base_model

PROMPT = """Summarize:
The Apollo program was the third United States human spaceflight program carried out by NASA,
which succeeded in landing the first humans on the Moon from 1969 to 1972.
Summary:
"""

def generate(model, tokenizer, label):
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False
        )
    print(f"\n--- {label} ---")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def main():
    base_model, tokenizer = load_base_model()

    # BEFORE LoRA
    generate(base_model, tokenizer, "Base Model Output")

    # AFTER LoRA
    lora_model = PeftModel.from_pretrained(
        base_model,
        "experiments/lora_summarization"
    )

    generate(lora_model, tokenizer, "With LoRA Adapter")


if __name__ == "__main__":
    main()

