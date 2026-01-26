from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    instructions="Speak like a detective.", # System message
    input="Why is Python such a popular language?", # User message
)

print(response.output_text)