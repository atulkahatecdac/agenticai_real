# pip install python-dotenv openai-agents

import asyncio
from agents import Agent, Runner
from dotenv import load_dotenv

load_dotenv(override=True)

# ---- Simple retriever (mock) ----
def retrieve_docs(query: str) -> str:
    return """
LoRA (Low-Rank Adaptation) reduces trainable parameters
by injecting low-rank matrices into attention layers.
"""

# ---- Fixed RAG pipeline ----
async def traditional_rag():
    docs = retrieve_docs("Explain LoRA")

    agent = Agent(
        name="RAGAgent",
        instructions="Answer the question using the provided context.",
        model="gpt-4o-mini"
    )

    prompt = f"""
Context:
{docs}

Question:
Explain LoRA in simple terms.
"""

    result = await Runner.run(agent, prompt)
    print("TRADITIONAL RAG OUTPUT:\n", result.final_output)

if __name__ == "__main__":
    asyncio.run(traditional_rag())
