import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_base_model(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="mps"
    )

    model.eval()
    return model, tokenizer
