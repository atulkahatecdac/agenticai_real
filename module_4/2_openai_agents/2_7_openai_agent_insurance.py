# pip install gradio pypdf python-dotenv openai chromadb openai-agents

import os
import gradio as gr
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, Runner, function_tool
import asyncio
import chromadb
from chromadb.utils import embedding_functions

# --- Load environment ---
load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Load PDF ---
PDF_PATH = "c://code//agenticai//2_openai_agents//Introduction_to_Insurance.pdf"
reader = PdfReader(PDF_PATH)
pdf_text = "".join([page.extract_text() or "" for page in reader.pages])

# --- Split PDF text into chunks ---
def split_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

pdf_chunks = split_text(pdf_text)

# --- ChromaDB persistent setup with Hugging Face embeddings ---
chroma_client = chromadb.PersistentClient(
    path=r"c:/code/agenticai/2_openai_agents/rag/chroma_pdf"
)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection_name = "pdf_collection"
existing_collections = [c.name for c in chroma_client.list_collections()]
if collection_name in existing_collections:
    collection = chroma_client.get_collection(name=collection_name)
    print(f"Loaded existing collection '{collection_name}'")
else:
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )
    print(f"Created new collection '{collection_name}'")
    # Add PDF chunks only if collection is new
    collection.add(
        documents=pdf_chunks,
        ids=[f"chunk_{i}" for i in range(len(pdf_chunks))]
    )
    print(f"ChromaDB populated with {len(pdf_chunks)} PDF chunks and embeddings")

# --- RAG Tool using Chroma semantic search ---
@function_tool
async def get_pdf_answer(query: str) -> str:
    result = collection.query(query_texts=[query], n_results=1)
    print("Top match details from ChromaDB:", result)  # Print top match details
    if result["documents"] and result["documents"][0]:
        chunk_text = result["documents"][0][0]
        prompt = (
            f"You are a helpful assistant answering questions about insurance policies.\n\n"
            f"Use only the following document content to answer:\n{chunk_text}\n\n"
            f"User question: {query}\n\n"
            f"Provide a concise and accurate answer:"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    return "I'm sorry, I couldn't find information related to that query."

# --- Agent ---
rag_agent = Agent(
    name="Customer Support RAG Bot",
    instructions=(
        "You are a helpful customer support assistant. "
        "Answer questions using the PDF content via your RAG tool."
    ),
    tools=[get_pdf_answer]
)

# --- Async chat handler ---
async def chat_with_rag(message, chat_history):
    session = await Runner.run(rag_agent, message)
    chat_history = chat_history or []
    chat_history.append((message, session.final_output))
    return chat_history, chat_history

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# Customer Support RAG Bot (PDF + Chroma + Hugging Face + GPT-4o-mini)")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a question about New India Assurance policies...")
    clear = gr.Button("Clear")

    async def respond(user_message, chat_history):
        return await chat_with_rag(user_message, chat_history)

    msg.submit(respond, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch()
