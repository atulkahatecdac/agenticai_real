# pip install python-dotenv openai-agents

import asyncio
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv

load_dotenv(override=True)

# ---- Retrieval tool ----
@function_tool
def search_knowledge(query: str) -> str:
    if "LoRA" in query:
        return "LoRA uses low-rank matrices to fine-tune large models efficiently."
    if "benefits" in query:
        return "Benefits: fewer parameters, faster training, lower memory usage."
    return "No relevant data found."

# ---- Agentic RAG agent ----
agent = Agent(
    name="AgenticRAGAgent",
    instructions="""
You are a research agent.
Decide when to retrieve information.
You may retrieve multiple times before answering.
""",
    tools=[search_knowledge],
    model="gpt-4o-mini"
)

async def agentic_rag():
    result = await Runner.run(
        agent,
        "Explain LoRA and its benefits"
    )
    print("AGENTIC RAG OUTPUT:\n", result.final_output)

if __name__ == "__main__":
    asyncio.run(agentic_rag())
