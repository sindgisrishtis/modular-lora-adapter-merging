import torch
from peft import PeftModel
from src.models.load_model import load_base_model


def get_layer_id(param_name):
    """
    Extract transformer layer index from parameter name.
    Example:
    base_model.model.layers.12.self_attn.q_proj.lora_A.weight
    """
    for token in param_name.split("."):
        if token.isdigit():
            return int(token)
    return None


def layerwise_merge(base_model, adapter_map):
    """
    adapter_map: list of tuples
    (adapter_path, layer_start, layer_end)
    """
    merged_state = {}

    for adapter_path, start, end in adapter_map:
        adapter_model = PeftModel.from_pretrained(base_model, adapter_path)
        state_dict = adapter_model.state_dict()

        for name, value in state_dict.items():
            if "lora_" not in name:
                continue

            layer_id = get_layer_id(name)
            if layer_id is None:
                continue

            if start <= layer_id <= end:
                merged_state[name] = value.clone()

    return merged_state


def main():
    base_model, _ = load_base_model()

    adapter_map = [
        ("experiments/lora_classification", 0, 10),
        ("experiments/lora_qa", 11, 21),
        ("experiments/lora_summarization", 22, 31),
    ]

    merged_state = layerwise_merge(base_model, adapter_map)

    torch.save(
        merged_state,
        "experiments/merged_layerwise_adapter.pt"
    )

    print("Layer-wise merged adapter saved successfully.")


if __name__ == "__main__":
    main()

