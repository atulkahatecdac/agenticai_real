from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI # Or any other LLM like Ollama

from dotenv import load_dotenv
load_dotenv(override=True)

# 1. Define the components
prompt = ChatPromptTemplate.from_template("Tell me a brief joke about {topic}")  # Automatically implements Runnable
model = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

# 2. The LCEL "Pipe"
# The output of prompt flows into model, which flows into the parser
chain = prompt | model | output_parser

# 3. Use the chain
# .invoke() is the standard way to run a Runnable
result = chain.invoke({"topic": "artificial intelligence"})

print(result)
