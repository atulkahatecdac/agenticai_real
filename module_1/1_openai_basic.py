# pip install openai dotenv
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv(override=True)

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Explain agentic AI in one paragraph"}
    ]
)

print(response.choices[0].message.content)
