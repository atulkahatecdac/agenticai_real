import os
import requests
import pandas as pd
import gradio as gr
from typing import TypedDict
from dotenv import load_dotenv
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

class ChatState(TypedDict):
    query: str
    answer:r str

data_path = r"C:\\code\agenticai_realpage\module_3\Dataset_banking_chatbot.csv"
df = pd.read_csv(data_path, encoding="latin-1")

texts = df["Query"].astype(str).tolist()
metadatas = df[["Response"]].to_dict(orient="records")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory=r"C:\code\agenticai_realpage\module_3\banking_chromadb",
    embedding_function=embeddings,
    collection_name="banking_faqs"
)

if vectordb._collection.count() == 0:
    vectordb.add_texts(texts=texts, metadatas=metadatas)
    vectordb.persist()

# Created a new topic in notify.sh named atulkahate_urgent_tickets
ntfy_topic = os.getenv("NTFY_TOPIC")
ntfy_url = f"https://ntfy.sh/{ntfy_topic}"

def send_ntfy(message):
    requests.post(ntfy_url, data=message.encode("utf-8"))

# -------------------------------
# Vector based intent detection
# Is the user asking to open a new bank account?
# If yes, then send a notification to notify.sh
# -------------------------------
ACCOUNT_OPEN_INTENT = "open a new bank account"

# Take user's query and check with a threshold for vector similarity
def is_account_opening(query: str, threshold: float = 0.7) -> bool:
    # embed_query converts text to a vector of ~384 dimensions
    query_vec = embeddings.embed_query(query)
    # Do the same for the intent string ("open a new bank account")
    intent_vec = embeddings.embed_query(ACCOUNT_OPEN_INTENT)

    # We have got lists of vectors. Convert them to numpy arrays
    # NumPy is needed for dot product, vector length
    query_vec = np.array(query_vec)
    intent_vec = np.array(intent_vec)
    
    # Cosine similarity formula: https://en.wikipedia.org/wiki/Cosine_similarity
    similarity = np.dot(query_vec, intent_vec) / (
        np.linalg.norm(query_vec) * np.linalg.norm(intent_vec)
    )

    # Compare the similarity with a threshold
    return similarity >= threshold

# -------------------------------
# Loan intent (EMAIL)
# -------------------------------
LOAN_INTENT = "apply for a bank loan"

def is_loan_request(query: str, threshold: float = 0.7) -> bool:
    query_vec = embeddings.embed_query(query)
    intent_vec = embeddings.embed_query(LOAN_INTENT)

    query_vec = np.array(query_vec)
    intent_vec = np.array(intent_vec)

    similarity = np.dot(query_vec, intent_vec) / (
        np.linalg.norm(query_vec) * np.linalg.norm(intent_vec)
    )
    
    print(f"***Debug*** Similarity: {similarity} ")

    return similarity >= threshold

# Send an email if user wants to apply for a loan
def send_email(message: str) -> str:
    from_email = "ekahate@gmail.com"  # Our Gmail address
    app_password = os.getenv("GMAIL_APP_PASSWORD")
    
    to_email = "newdelthis@gmail.com"
    subject = "Loan Application Request"
    body = f"Customer wants to apply for a loan:\n\n{message}"

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, app_password)
        server.send_message(msg)
        server.quit()
        print("***Debug*** Email sent successfully")
        return "Email sent successfully"
    except Exception as e:
        print(f"***Debug*** Email failed: {str(e)}")
        return f"Email send failed: {str(e)}"


# -------------------------------
# Banking node
# -------------------------------
def banking_node(state: ChatState) -> ChatState:
    results = vectordb.similarity_search(
        state["query"],
        k=1
    )

    if not results:
        state["answer"] = "Sorry I could not find the answer"
        return state

    doc = results[0]
    response = doc.metadata["Response"]

    if is_account_opening(state["query"]):
        message = (
            "New bank account request\n\n"
            + "User query\n"
            + state["query"]
            + "\n\n"
            + "System response\n"
            + response
        )
        send_ntfy(message)

    if is_loan_request(state["query"]):
        send_email(state["query"])

    state["answer"] = response
    return state

graph = StateGraph(ChatState)
graph.add_node("banking", banking_node)
graph.set_entry_point("banking")
graph.add_edge("banking", END)

app = graph.compile()

def chat_fn(message, history):
    return app.invoke({"query": message})["answer"]

demo = gr.ChatInterface(
    fn=chat_fn,
    title="Banking Assistant",
    examples=[
        "I want to open a new bank account",
        "I want to apply for a home loan",
        "How can I check my account balance"
    ]
)

if __name__ == "__main__":
    demo.launch()
