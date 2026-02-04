import torch
from src.models.load_model import load_base_model


def main():
    model, tokenizer = load_base_model()

    merged_state = torch.load("experiments/merged_layerwise_adapter.pt")
    model.load_state_dict(merged_state, strict=False)

    prompts = [
        "Summarize: Artificial intelligence is transforming industries.",
        "Context: Paris is in France.\nQuestion: Is Paris in Europe?\nAnswer:",
        "Text: Apple shares rose after strong earnings.\nLabel:"
    ]

    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=20)
        print("\nPrompt:", p)
        print("Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

