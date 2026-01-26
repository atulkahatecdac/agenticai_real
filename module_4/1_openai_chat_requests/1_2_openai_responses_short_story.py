# Basic example to ask a model to generate a short story

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI()

input="Write a short note on the importance pf Kubernetes."

response = client.responses.create(
    model="gpt-4o-mini",
    input=input
)

# print(response)

# From OpenAI documentation: Some of our official SDKs include an output_text property 
#   on model responses for convenience, which aggregates all text outputs from the model 
#   into a single string. This may be useful as a shortcut to 
#   access text output from the model.
print(response.output_text)