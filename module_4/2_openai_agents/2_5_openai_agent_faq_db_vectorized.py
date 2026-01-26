# pip install python-dotenv sentence-transformers chromadb openai-agents

from dotenv import load_dotenv
import os
import sqlite3
import asyncio

from sentence_transformers import SentenceTransformer
from agents import Agent, Runner, function_tool

import chromadb
from chromadb.utils import embedding_functions


# ----------------------------------------------------
# 1. Load environment
# ----------------------------------------------------
load_dotenv(override=True)


# ----------------------------------------------------
# 2. Load FAQs from SQLite
# ----------------------------------------------------
DB_PATH = r"c://code//agenticai//2_openai_agents//faqs.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

def load_faqs():
    cursor.execute("SELECT topic, answer FROM faqs")
    return dict(cursor.fetchall())

knowledge_base = load_faqs()
print(f"Loaded {len(knowledge_base)} FAQs from SQLite")


# ----------------------------------------------------
# 3. Setup ChromaDB (local persistent)
# ----------------------------------------------------
chroma_client = chromadb.PersistentClient(
    path=r"c:/code/agenticai/2_openai_agents/rag/chromadb"
)

# HuggingFace embedding model
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create collection
collection = chroma_client.get_or_create_collection(
    name="faq_collection",
    embedding_function=embedding_fn
)


# ----------------------------------------------------
# 4. Insert SQLite FAQs into ChromaDB
# ----------------------------------------------------
# Clear old docs to avoid duplicates
# Safely remove all entries
existing = collection.get()

if existing["ids"]:
    collection.delete(ids=existing["ids"])

collection.add(
    documents=list(knowledge_base.values()),
    ids=list(knowledge_base.keys()),
)

print("ChromaDB populated with FAQ documents")


# ----------------------------------------------------
# 5. Tool: Query ChromaDB (no SQL, no cosine_calc)
# ----------------------------------------------------
@function_tool
async def get_faq_answer(query: str) -> str:
    """
    Perform semantic search on the locally stored ChromaDB.
    Returns the closest FAQ answer.
    """

    result = collection.query(
        query_texts=[query],
        n_results=1
    )

    if result["documents"] and len(result["documents"][0]) > 0:
        return result["documents"][0][0]  # top match
    
    return "Sorry, I couldn't find information about that topic."


# ----------------------------------------------------
# 6. Agent
# ----------------------------------------------------
faq_agent = Agent(
    name="Customer Support Bot",
    instructions="You are a friendly support assistant. Use the FAQ search tool.",
    tools=[get_faq_answer],
)


async def chat_with_support(message):
    session = await Runner.run(faq_agent, message)
    return session.final_output


# ----------------------------------------------------
# 7. Interactive loop
# ----------------------------------------------------
async def main():
    print("Customer Support Bot running. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting.")
            break

        response = await chat_with_support(user_input)
        print("Bot:", response)


if __name__ == "__main__":
    asyncio.run(main())
