import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from src.models.load_model import load_base_model


def merge_adapters(
    base_model,
    adapter_paths,
    weights
):
    assert len(adapter_paths) == len(weights)

    merged_state = {}

    for path, weight in zip(adapter_paths, weights):
        model = PeftModel.from_pretrained(base_model, path)
        state_dict = model.state_dict()

        for k, v in state_dict.items():
            if "lora_" in k:
                if k not in merged_state:
                    merged_state[k] = weight * v
                else:
                    merged_state[k] += weight * v

    return merged_state


def save_merged_adapter(merged_state, save_path):
    torch.save(merged_state, save_path)


def main():
    base_model, _ = load_base_model()

    adapter_paths = [
        "experiments/lora_summarization",
        "experiments/lora_qa",
        "experiments/lora_classification"
    ]

    weights = [0.4, 0.3, 0.3]

    merged_state = merge_adapters(
        base_model,
        adapter_paths,
        weights
    )

    save_merged_adapter(
        merged_state,
        "experiments/merged_weighted_adapter.pt"
    )

    print("Merged adapter saved successfully.")


if __name__ == "__main__":
    main()

