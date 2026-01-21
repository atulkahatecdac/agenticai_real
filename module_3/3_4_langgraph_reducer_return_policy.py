from typing import TypedDict, List
from typing_extensions import Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv(override=True)

# -----------------------------
# 1) LLM
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -----------------------------
# 2) Reducer
# -----------------------------
def merge_results(old: List[str], new: List[str]) -> List[str]:
    return old + new

# -----------------------------
# 3) State
# -----------------------------
class State(TypedDict):
    amazon_url: str
    flipkart_url: str
    results: Annotated[List[str], merge_results]

# -----------------------------
# 4) Parallel LLM agents
# -----------------------------
def amazon_agent(state: State):
    prompt = f"""
    Summarize the key points of Amazon India's return policy:
    {state['amazon_url']}

    Focus on:
    - Return window
    - Refund vs replacement
    - Conditions
    """
    resp = llm.invoke(prompt)
    return {"results": [f"[Amazon]\n{resp.content}"]}

def flipkart_agent(state: State):
    prompt = f"""
    Summarize the key points of Flipkart's return policy:
    {state['flipkart_url']}

    Focus on:
    - Return / replacement window
    - Category-based rules
    - Pickup & refund conditions
    """
    resp = llm.invoke(prompt)
    return {"results": [f"[Flipkart]\n{resp.content}"]}

# -----------------------------
# 5) Graph (CORRECT parallel fan-out)
# -----------------------------
graph = StateGraph(State)

graph.add_node("amazon", amazon_agent)
graph.add_node("flipkart", flipkart_agent)

# Entry point
graph.set_entry_point("amazon")

# Fan-out to BOTH agents
graph.add_edge("amazon", "flipkart")
graph.add_edge("amazon", END)
graph.add_edge("flipkart", END)

app = graph.compile()

# -----------------------------
# 6) Run
# -----------------------------
initial_state = {
    "amazon_url": "https://www.amazon.in/gp/help/customer/display.html?nodeId=202111910",
    "flipkart_url": "https://www.flipkart.com/pages/returnpolicy",
    "results": []
}

result = app.invoke(initial_state)

print("\n--- Comparison ---\n")
for r in result["results"]:
    print(r)
    print("-" * 40)
