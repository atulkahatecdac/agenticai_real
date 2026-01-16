# pip install "langserve[all]" fastapi uvicorn pydantic

from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

from dotenv import load_dotenv
load_dotenv(override=True)

# 1. Define the Input Schema explicitly
# This tells LangServe/Playground exactly what variables to expect
class InputSchema(BaseModel):
    topic: str = Field(..., description="The subject to summarize")

# 2. Create the FastAPI app
app = FastAPI(
    title="My LangChain Server",
    version="1.0",
    description="A simple API server for my AI chain"
)

# 3. Define your chain
model = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("Write a 5-point summary about {topic}")
chain = prompt | model

# 4. Add the routes with the custom types
add_routes(
    app,
    # We use .with_types to bind our Pydantic model to the chain
    chain.with_types(input_type=InputSchema),
    path="/note",
)

# 5. Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, log_level="info")