from peft import LoraConfig, TaskType

def get_lora_config(
    rank=8,
    alpha=16,
    dropout=0.05
):
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

