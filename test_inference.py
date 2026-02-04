import torch
from src.models.load_model import load_base_model

model, tokenizer = load_base_model()

prompt = "Explain LoRA in simple terms."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

