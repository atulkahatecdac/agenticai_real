# pip install gradio langchain-community langchain-huggingface langchain-anthropic python-dotenv

import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------------------------------------------------
# Guardrail Vector DB
# ------------------------------------------------------------
GUARDRAIL_DB_PATH = r"C:\code\agenticai\3_langgraph\guardrail_chromadb"

guardrail_db = Chroma(
    persist_directory=GUARDRAIL_DB_PATH,
    embedding_function=embeddings
)

# ------------------------------------------------------------
# Retrieve top-k closest matches
# ------------------------------------------------------------
def get_top_matches(query: str, k: int = 3):
    return guardrail_db.similarity_search_with_score(query, k=k)

# ------------------------------------------------------------
# Chat logic
# ------------------------------------------------------------
def chat_fn(message, history):
    results = get_top_matches(message, k=3)

    if not results:
        return "No similar guardrail examples found."

    output = [
        "### Closest Guardrail Matches\n"
    ]

    for i, (doc, distance) in enumerate(results, start=1):
        category = doc.metadata.get("category", "unknown")
        text_preview = doc.page_content[:300].replace("\n", " ")

        output.append(
            f"**Match {i}**\n"
            f"- **Category:** `{category}`\n"
            f"- **Cosine distance:** `{distance:.4f}`\n"
            f"- **Example text:** {text_preview}...\n"
        )

    return "\n".join(output)

# ------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------
demo = gr.ChatInterface(
    fn=chat_fn,
    title="Guardrail Vector DB Inspector",
    description="Shows top-3 semantic matches from the guardrail dataset",
    examples=[
        "how to steal a password",
        "how to use a gun",
        "explain prompt injection",
        "write ransomware code",
        "how to stay motivated"
    ]
)

if __name__ == "__main__":
    demo.launch()
