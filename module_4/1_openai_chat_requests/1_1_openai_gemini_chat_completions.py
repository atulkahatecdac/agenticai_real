# pip install google-generativeai python-dotenv

import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# Configure Gemini with our API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ----------------------------
# First basic question
# ----------------------------
model = genai.GenerativeModel("gemini-2.5-flash")

prompt = "Tell me three good points about machine learning."
response = model.generate_content(prompt)

print(response.text)

# ----------------------------
# Now a tougher question
# ----------------------------
question_prompt = "Please propose a hard, challenging question to assess someone's IQ. Respond only with the question."
response = model.generate_content(question_prompt)

question = response.text
print(f"Gemini Question: {question}")

# ----------------------------
# Ask Gemini to answer its own question
# ----------------------------
response = model.generate_content(question)
answer = response.text

print(f"Gemini Answer: {answer}")
