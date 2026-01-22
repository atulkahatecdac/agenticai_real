# pip install langchain-community langgraph pandas gradio chromadb

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from typing import TypedDict
import pandas as pd
import gradio as gr
import chromadb
from chromadb.config import Settings

# --- Step 1: Define state ---
class ChatState(TypedDict):
    query: str
    answer: str

# --- Step 2: Load dataset ---
# Original data source: https://www.kaggle.com/datasets/manojajj/banking-chatbot
csv_path = "c://code//agenticai//3_langgraph//Dataset_banking_chatbot.csv"
df = pd.read_csv(csv_path, encoding='latin-1')

texts = df["Query"].astype(str).tolist()
metadatas = df[["Query", "Response"]].to_dict(orient="records")

# --- Step 3: Embeddings ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- Step 4: Setup Chroma ---
persist_dir = "c://code//agenticai//3_langgraph//banking_chromadb"
collection_name = "banking_faqs"

client = chromadb.PersistentClient(
    path=persist_dir,
    settings=Settings()
)

existing_collections = [c.name for c in client.list_collections()]

if collection_name in existing_collections:
    print(f"Loading existing collection '{collection_name}'...")
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name
    )
else:
    print(f"Creating new collection '{collection_name}'...")
    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    vectordb.persist()

# --- Step 5: LangGraph node ---
def retrieve_answer(state: ChatState) -> ChatState:
    results = vectordb.similarity_search_with_score(
        state["query"], k=1
    )

    if not results:
        state["answer"] = "Sorry, I could not find an answer."
        return state

    doc, distance = results[0]
    similarity = 1 - distance

    response = doc.metadata["Response"]

    state["answer"] = (
        f"{response}\n\n"
        f"(Confidence: {similarity:.2f})"
    )
    return state

# --- Step 6: Build LangGraph ---
graph = StateGraph(ChatState)
graph.add_node("retrieve", retrieve_answer)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", END)
runnable = graph.compile()

# --- Step 7: Gradio handler ---
def chat_fn(message, history):
    return runnable.invoke({"query": message})["answer"]

# --- Step 8: Gradio UI ---
demo = gr.ChatInterface(
    fn=chat_fn,
    title="Banking FAQ Chatbot (Vector Search)",
    examples=[
        "How can I open a new bank account?",
        "What documents are required to open an account?",
        "How do I check my balance?"
    ],
)

if __name__ == "__main__":
    demo.launch()
