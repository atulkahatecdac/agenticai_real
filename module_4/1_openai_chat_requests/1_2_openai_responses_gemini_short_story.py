# pip install google-generativeai python-dotenv

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv(override=True)

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Create a Gemini model client (use flash for speed or pro for accuracy)
model = genai.GenerativeModel("gemini-2.0-flash")

# Example prompt
prompt = "Write a short story of three lines about an AI Agent who wanted to learn singing."

# Generate response
response = model.generate_content(prompt)

# Print the full response text (similar to response.output_text in OpenAI)
print(response.text)
