from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "microsoft/phi-1_5"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu"
)

inputs = tokenizer(
    "Explain AI transformers in one short paragraph. Do not give exercises or examples.",
    return_tensors="pt"
)

outputs = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=False
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
