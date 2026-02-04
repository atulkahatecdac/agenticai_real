# pip install openai python-dotenv chromadb openai-agents ragas datasets sentence-transformers

from openai import OpenAI
from dotenv import load_dotenv
import asyncio
from agents import Agent, Runner, function_tool
import chromadb
from chromadb.utils import embedding_functions

# --- RAGAS imports ---
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision
)

from ragas.embeddings import HuggingFaceEmbeddings

load_dotenv(override=True)

client = OpenAI()

# ----------------------------------------------------
# Global variable to store retrieved context
# ----------------------------------------------------
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
# 2. Insert knowledge base (Avoid duplicate inserts)
# ----------------------------------------------------
knowledge_base = {
    "shipping_time": "Our standard shipping time is 3-5 business days.",
    "return_policy": "You can return any product within 30 days of delivery.",
    "warranty": "All products come with a one-year warranty covering manufacturing defects.",
    "payment_methods": "We accept credit cards, debit cards, and PayPal.",
    "customer_support": "You can reach our support team 24/7 via email or chat."
}

existing_ids = set(collection.get()["ids"])

new_ids = []
new_docs = []

for k, v in knowledge_base.items():
    if k not in existing_ids:
        new_ids.append(k)
        new_docs.append(v)

if new_docs:
    collection.add(documents=new_docs, ids=new_ids)

# ----------------------------------------------------
# 3. Tool used by Agent (RAG Retrieval)
# ----------------------------------------------------
@function_tool
async def faq_invoker(topic: str) -> str:
    global last_retrieved_context

    result = collection.query(query_texts=[topic], n_results=2)

    docs = result["documents"][0] if result["documents"] else []
    
    # Example: last_retrieved_context = ["You can return any product within 30 days of delivery."]
    last_retrieved_context = docs

    if docs:
        return docs[0]

    return "Sorry, I couldn't find information about that."

# ----------------------------------------------------
# 4. Agent Definition
# ----------------------------------------------------
faq_agent = Agent(
    name="Customer Support Bot",
    instructions=(
        "You are a helpful customer support assistant. "
        "Use the FAQ search tool when appropriate."
    ),
    tools=[faq_invoker]
)

# ----------------------------------------------------
# 5. Setup RAGAS HuggingFace Embeddings (Modern)
# ----------------------------------------------------
ragas_embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------------------------------
# 6. RAGAS Evaluation Function
# ----------------------------------------------------
def evaluate_rag(question, answer, contexts):

    # Using the top retrieved context as a proxy reference for evaluation
    reference = contexts[0] if contexts else ""

    data = {
        "question": [question],   # What the user asked
        "answer": [answer],       # What the agent produced
        "contexts": [contexts],   # What the agent retrieved
        "reference": [reference]  # The reference answer (Proxy ground truth from retrieval)
    }

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision()
        ],
        embeddings=ragas_embeddings
    )

    return result



# ----------------------------------------------------
# 7. Chat Handler
# ----------------------------------------------------
async def chat_with_support(message):
    global last_retrieved_context

    session = await Runner.run(faq_agent, message)
    
    # Agent produces the final answer, e.g. "You can return any product within 30 days of delivery."
    # No evaluation yet, just the normal RAG inference
    answer = session.final_output

    # Now evaluate (post-hoc RAGAS)
    if last_retrieved_context:
        scores = evaluate_rag(
            question=message, 
            answer=answer, 
            contexts=last_retrieved_context 
        )

        print("\nRAGAS Scores:")
        print(scores)
        
        # Answer relevancy might be printed as NaN
        # It asks: “How well does the answer address the question?”
        # Extracts key intents / entities from the question
        # Checks whether those are present in the answer
        # If it cannot extract anything meaningful, it returns NaN
        # So, NAN is not bad, it just means I do not have enough context to answer the question
        # Example: Suppose our question is "Return" or "Return policy"
        # Actual answer is more detailed
        # RAGAS cannot extract a clear intent, so it returns NaN
        # Generally happens with keyword-style questions

    return answer

# ----------------------------------------------------
# 8. Interactive Loop
# ----------------------------------------------------
async def main():
    print("Customer Support Bot is running. Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            break

        response = await chat_with_support(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    asyncio.run(main())
