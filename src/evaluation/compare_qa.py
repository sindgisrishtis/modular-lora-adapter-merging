import torch
from peft import PeftModel

from src.models.load_model import load_base_model
PROMPT = """Context:
The statement "Paris is the capital of Germany" is incorrect.
Question:
Is Paris the capital of Germany?
Answer:
"""


def generate(model, tokenizer, label):
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False
        )
    print(f"\n--- {label} ---")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def main():
    base_model, tokenizer = load_base_model()

    generate(base_model, tokenizer, "Base Model Output")

    qa_model = PeftModel.from_pretrained(
        base_model,
        "experiments/lora_qa"
    )

    generate(qa_model, tokenizer, "With QA LoRA Adapter")


if __name__ == "__main__":
    main()

