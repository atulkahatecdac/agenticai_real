# pip install openai python-dotenv chromadb openai-agents

from openai import OpenAI
from dotenv import load_dotenv
import asyncio
from agents import Agent, Runner, function_tool
import chromadb
from chromadb.utils import embedding_functions

load_dotenv(override=True)

client = OpenAI()

# ----------------------------------------------------
# 1. Initialize persistent ChromaDB
# ----------------------------------------------------
chroma_client = chromadb.PersistentClient(
    path=r"c:/code/agenticai/2_openai_agents/rag/chromadb"
)

# HuggingFace embeddings
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create / load collection
collection = chroma_client.get_or_create_collection(
    name="chromadb_faq_support",
    embedding_function=embedding_fn
)

# ----------------------------------------------------
# 2. Insert knowledge base into ChromaDB
# ----------------------------------------------------
knowledge_base = {
    "shipping_time": "Our standard shipping time is 3-5 business days.",
    "return_policy": "You can return any product within 30 days of delivery.",
    "warranty": "All products come with a one-year warranty covering manufacturing defects.",
    "payment_methods": "We accept credit cards, debit cards, and PayPal.",
    "customer_support": "You can reach our support team 24/7 via email or chat."
}

collection.add(
    documents=list(knowledge_base.values()),
    ids=list(knowledge_base.keys())
)

# ----------------------------------------------------
# 3. Tool used by Agent to query ChromaDB (RAG)
# ----------------------------------------------------
@function_tool
async def faq_invoker(topic: str) -> str:
    """Answer FAQs using Chroma semantic search."""
    
    result = collection.query(query_texts=[topic], n_results=1)

    if result["documents"] and len(result["documents"][0]) > 0:
        doc = result["documents"][0][0]   # first result, first document
    else:
        doc = None

    return doc or "Sorry, I couldn't find information about that."


# ----------------------------------------------------
# 4. Define the Agent
# ----------------------------------------------------
faq_agent = Agent(
    name="Customer Support Bot",
    instructions=(
        "You are a helpful customer support assistant. "
        "Use the FAQ search tool when appropriate."
    ),
    tools=[faq_invoker]
)

async def chat_with_support(message):
    session = await Runner.run(faq_agent, message)
    return session.final_output

# ----------------------------------------------------
# 5. Simple interactive loop
# ----------------------------------------------------
async def main():
    print("Customer Support Bot is running. Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting.")
            break
        
        response = await chat_with_support(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    asyncio.run(main())