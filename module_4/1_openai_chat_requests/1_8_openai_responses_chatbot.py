# pip install pypdf gradio
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import gradio as gr

load_dotenv(override=True)

client = OpenAI()

# --- Step 1: Read Buffett PDF into a single string ---
reader = PdfReader("C://code//agenticai//1_openai_chat_requests//Warren_Buffett.pdf")
buffett = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        buffett += text

# --- Step 2: Define chat function ---
def chat_with_buffett(message, history):
    """
    Chat with Buffett knowledge base using OpenAI Responses API.
    message: user input
    history: previous chat messages (not needed for stateless, but Gradio passes it)
    """
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            # Pass only the first 4k of Buffett words, ~ 1000 tokens
            # Avoids overloading: Earlier models supported just 4k tokens, GPT-4o-mini supports ~128k
            {"role": "system", "content": f"You are a helpful assistant. You can answer based on the following text:\n\n{buffett[:4000]} or from the Internet."},
            {"role": "user", "content": message},
        ]
    )
    return response.output_text

# --- Step 3: Build Gradio UI ---

# Blocks is a layout system in Gradio, more advanced than 
#  gradio.Interface
# With creates a context manager, so that all components defined next
#  will get added to this demo app
with gr.Blocks() as demo: # start defining the UI
    gr.Markdown("#Ask about Warren Buffett") # means big header
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a question about Buffett...")
    clear = gr.Button("Clear")
    
    def respond(user_message, chat_history):
        answer = chat_with_buffett(user_message, chat_history)
        chat_history.append((user_message, answer))
        return "", chat_history

    # Syntax: msg.submit(function, inputs, outputs)
    # When the user clicks ENTER, pass inputs [msg, chatbot] to the chat function
    # Then take the function's output and add it to the chat history [msg, chatbot]
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    # When the user clicks the "Clear" button, clear the chat history
    # Run the function lambda: None, which means do nothing
    # Function takes no inputs (None) and sends its output (None) to the chatbot
    # This resets the chat history
    # queue=False means run the function immediately
    clear.click(lambda: None, None, chatbot, queue=False)
    
demo.launch()
