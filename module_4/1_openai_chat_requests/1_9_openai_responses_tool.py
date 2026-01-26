from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader

# -----------------------------
# 1. Setup and load knowledge base
# -----------------------------
load_dotenv(override=True)
client = OpenAI()

reader = PdfReader("C://code//agenticai//1_openai_chat_requests//Warren_Buffett.pdf")
buffett_text = "".join(
    page.extract_text() or "" for page in reader.pages
)

# -----------------------------
# 2. Define a simple callable tool
# -----------------------------
def summarize_text(text: str, length: str = "short") -> str:
    """Summarize text using GPT."""
    summary = client.responses.create(
        model="gpt-4o-mini",
        input=f"Summarize this text in a {length} way:\n\n{text[:2000]}"
    )
    return summary.output_text

list_tools = [
    {
        "name": "summarize_text",
        "type": "function",
        "function": {
            "name": "summarize_text",
            "description": "Summarize the given text concisely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "length": {
                        "type": "string",
                        "enum": ["short", "medium", "long"]
                    }
                },
                "required": ["text"]
            }
        }
    }
]

# -----------------------------
# 3. Core agent logic
# -----------------------------
def ask_agent(prompt: str):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "You are an intelligent agent trained on Warren Buffett's writings. "
                    "If the question is complex or needs context, call the summarize_text tool."
                )
            },
            {
                # System: You are Warren Buffett’s AI assistant. Use tools if needed.
                # User: How does Buffett view diversification? (Here’s a chunk of his text)
                "role": "user",
                "content": f"{prompt}\n\nReference text:\n{buffett_text[:4000]}"
            }
        ],
        tools=list_tools
    )

    # If the model calls the summarize_text tool
    if hasattr(response.output[0], "type") and response.output[0].type == "tool_call":
        tool_call = response.output[0].tool_call
        if tool_call.function.name == "summarize_text":
            args = tool_call.function.arguments
            result = summarize_text(**args)

            # Give result back to the model for final answer
            follow_up = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"Tool result: {result}"}
                ]
            )
            return follow_up.output_text

    # If no tool call was made, return normal model output
    return response.output_text


# -----------------------------
# 4. Try it out interactively
# -----------------------------
if __name__ == "__main__":
    print("Warren Buffett Agent Ready!\n(Type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        answer = ask_agent(user_input)
        print(f"\nAgent: {answer}\n")
