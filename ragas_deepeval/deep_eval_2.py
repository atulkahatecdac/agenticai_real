import os
import asyncio
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
from chromadb.utils import embedding_functions

# --- DeepEval Imports ---
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric, 
    AnswerRelevancyMetric, 
    ContextualPrecisionMetric
)

load_dotenv(override=True)
client = OpenAI()

# Global variable to store retrieved context for evaluation
last_retrieved_context = []

# ----------------------------------------------------
# 1. Initialize persistent ChromaDB
# ----------------------------------------------------
chroma_client = chromadb.PersistentClient(
    path=r"c:/code/agenticai/2_openai_agents/rag/chromadb"
)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = chroma_client.get_or_create_collection(
    name="chromadb_faq_support",
    embedding_function=embedding_fn
)

# ----------------------------------------------------
# 2. Tool used by Agent (RAG Retrieval)
# ----------------------------------------------------
@function_tool
async def faq_invoker(topic: str) -> str:
    global last_retrieved_context
    result = collection.query(query_texts=[topic], n_results=2)
    docs = result["documents"][0] if result["documents"] else []
    
    # DeepEval expects context as a list of strings
    last_retrieved_context = docs

    return docs[0] if docs else "Sorry, I couldn't find information about that."

# ----------------------------------------------------
# 3. Agent Definition
# ----------------------------------------------------
faq_agent = Agent(
    name="Customer Support Bot",
    instructions="You are a helpful customer support assistant. Use the FAQ search tool.",
    tools=[faq_invoker]
)

# ----------------------------------------------------
# 4. DeepEval Evaluation Function
# ----------------------------------------------------
def evaluate_with_deepeval(question, answer, contexts):
    # Proxy ground truth from the first retrieved document
    reference = contexts[0] if contexts else ""

    # Create the test case object
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=contexts,
        expected_output=reference
    )

    # Initialize DeepEval Metrics
    # (Thresholds can be adjusted: 0.0 to 1.0)
    metrics = [
        FaithfulnessMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.7),
        ContextualPrecisionMetric(threshold=0.7)
    ]

    # Run evaluation
    # Set ignore_errors=True if you want the loop to continue on minor LLM hiccups
    evaluate(test_cases=[test_case], metrics=metrics)

# ----------------------------------------------------
# 5. Chat Handler
# ----------------------------------------------------
async def chat_with_support(message):
    global last_retrieved_context
    last_retrieved_context = [] # Reset for each new turn

    session = await Runner.run(faq_agent, message)
    answer = session.final_output

    if last_retrieved_context:
        print("\n--- Starting DeepEval Evaluation ---")
        evaluate_with_deepeval(
            question=message, 
            answer=answer, 
            contexts=last_retrieved_context 
        )
    
    return answer

# ----------------------------------------------------
# 6. Interactive Loop
# ----------------------------------------------------
async def main():
    print("Customer Support Bot (DeepEval Ready). Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit": break
        response = await chat_with_support(user_input)
        print(f"\nBot: {response}")

if __name__ == "__main__":
    asyncio.run(main())