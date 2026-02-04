from src.models.load_model import load_base_model

model, tokenizer = load_base_model()

print("Model loaded successfully")
print("Device:", model.device)

