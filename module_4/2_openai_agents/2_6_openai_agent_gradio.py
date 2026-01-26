# pip install python-dotenv gradio chromadb openai-agents

from dotenv import load_dotenv
import asyncio
import gradio as gr
from agents import Agent, Runner, function_tool
import chromadb
from chromadb.utils import embedding_functions

# --- Setup ---
load_dotenv(override=True)

# --- ChromaDB persistent setup with Hugging Face embeddings ---
chroma_client = chromadb.PersistentClient(
    path=r"c:/code/agenticai/2_openai_agents/rag/chromadb"
)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="faq_collection",
    embedding_function=embedding_fn
)

# --- Knowledge base ---
knowledge_base = {
    "shipping_time": "Our standard shipping time is 3-5 business days.",
    "return_policy": "You can return any product within 30 days of delivery.",
    "warranty": "All products come with a one-year warranty covering manufacturing defects.",
    "payment_methods": "We accept credit cards, debit cards, and PayPal.",
    "customer_support": "You can reach our support team 24/7 via email or chat."
}

# Clear old docs and add new ones
existing = collection.get()
if existing["ids"]:
    collection.delete(ids=existing["ids"])

collection.add(
    documents=list(knowledge_base.values()),
    ids=list(knowledge_base.keys()),
)
print("ChromaDB populated with FAQ documents and embeddings")

# --- FAQ Tool using Chroma semantic search ---
@function_tool
async def get_faq_answer(query: str) -> str:
    result = collection.query(query_texts=[query], n_results=1)
    if result["documents"] and result["documents"][0]:
        return result["documents"][0][0]
    return "Sorry, I couldn't find information about that topic."

# --- Agent ---
faq_agent = Agent(
    name="Customer Support Bot",
    instructions="You are a friendly support assistant. Use your FAQ tool to answer user questions.",
    tools=[get_faq_answer],
)

# Takes user message and chat history
async def chat_with_support(message, chat_history):
    # Execute the faq_agent on the user message
    # await is used because Runner.run() is an async function
    # and may take time to complete
    session = await Runner.run(faq_agent, message)
    # Ensure that chat_history is a list
    # If it was None (first run), make it an empty list
    chat_history = chat_history or []
    # Add a tuple of (user message, bot response) to the chat history
    chat_history.append((message, session.final_output))
    # Return two copies of the chat history
    # First copy is for Gradio chat window in UI
    # Second copy is for the chat_history variable,
    # passed back as chat_history to the next user turn
    return chat_history, chat_history

# --- Gradio UI ---

# gr.Blocks() creates a container for our UI components
with gr.Blocks() as demo:
    # gr.Markdown() displays text in Markdown format
    gr.Markdown("# Customer Support Bot")
    # gr.Chatbot() creates a chat window
    # It can display message pairs: (user input, bot response)
    chatbot = gr.Chatbot()
    # gr.Textbox() creates a text box for user input
    msg = gr.Textbox(placeholder="Ask a question about our products or policies...")
    clear = gr.Button("Clear")

    # The respond() function is triggered when the user submits a message
    async def respond(user_message, chat_history):
        # Call async chat_with_support() function to 
        # send the user message to the agent and get the response
        # returns updated chat history to chat window in real time
        return await chat_with_support(user_message, chat_history)

    # Connects the Textbox(msg) to the respond() function
    # [msg, chatbot] are inputs to the respond() function
    # [chatbot, chatbot] are outputs from the respond() function
    # they are added to the chat window
    # When the user clicks ENTER, the respond() function is triggered
    # and chat history is updated
    msg.submit(respond, [msg, chatbot], [chatbot, chatbot])
    
    # When the user clicks the "Clear" button, clear the chat history
    # by returning an empty list
    clear.click(lambda: [], None, chatbot)

demo.launch()
