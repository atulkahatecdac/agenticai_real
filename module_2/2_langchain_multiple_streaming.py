# Two separate chains, streaming output in parallel

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv(override=True)

# 1. Setup the Model
model = ChatOpenAI(model="gpt-4o-mini", streaming=True)

# 2. Define two separate chains
summary_chain = (
    ChatPromptTemplate.from_template("Summarize this text in 1 sentence: {text}") 
    | model 
    | StrOutputParser()
)

sentiment_chain = (
    ChatPromptTemplate.from_template("What is the sentiment of this text? {text}") 
    | model 
    | StrOutputParser()
)

# 3. Combine them using RunnableParallel
# This runs both chains at the exact same time
parallel_chain = RunnableParallel(
    summary=summary_chain,
    sentiment=sentiment_chain
)

# 4. Stream the results
input_text = "I absolutely love the new LangChain LCEL syntax! It makes parallelizing tasks so much easier than the old way."

print("Streaming Parallel Results:\n")
for chunk in parallel_chain.stream({"text": input_text}):
    print(chunk)