from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set")

client = InferenceClient(
    token=HF_TOKEN,
    provider="hf-inference"
)

response = client.text_generation(
    model="bigscience/bloomz-560m",
    prompt="Explain transformers in one short paragraph.",
    max_new_tokens=120
)

print(response)
