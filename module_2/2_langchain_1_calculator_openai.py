from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv(override=True)

# Load OpenAI model
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.8,
)

# Messages (same structure as Ollama)
messages = [
    SystemMessage(content="You are a math expert"),
    HumanMessage(content="What is the square root of 324 plus 45?"),
]

# Invoke model
response = model.invoke(messages)

print("Response:", response.content)
