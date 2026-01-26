# pip install ollama

import ollama

MODEL_NAME = "llama3.2:latest"

prompt = input="Write a short story of three lines about an AI Agent who wanted to learn singing."

# Equivalent to OpenAI's responses.create()
response = ollama.generate(
    model=MODEL_NAME,
    prompt=prompt
)

print(response['response'])
