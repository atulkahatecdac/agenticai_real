from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

model = ChatOllama(
    model="llama3.2:latest",
    temperature=0.8,
)

messages = [
    SystemMessage(content="You are a math expert"),
    HumanMessage(content="What is the square root of 324 plus 45?")
]

response = model.invoke(messages)
print("Response:", response)
