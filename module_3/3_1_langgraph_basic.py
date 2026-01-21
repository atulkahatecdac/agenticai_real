from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# --- Define the state structure ---
class InputState(TypedDict):
    name: str
    greeting: str

# --- Node 1: Ask for name ---
def ask_name(state: InputState) -> InputState:
    name = input("Bot: What's your name? ")
    return {"name": name, "greeting": ""}

# --- Node 2: Greet user ---
def greet_user(state: InputState) -> InputState:
    # If name is empty, use "there"
    name = state.get("name") or "there"
    greeting = f"Hello {name}, nice to meet you!"
    print("Bot:", greeting)
    return {"name": name, "greeting": greeting}

# --- Build the graph ---
def create_graph():
    graph = StateGraph(InputState)
    
    graph.add_node("branch_a", ask_name)
    graph.add_node("branch_b", greet_user)
    
    graph.add_edge(START, "branch_a")
    graph.add_edge("branch_a", "branch_b")
    graph.add_edge("branch_b", END)
    
    return graph.compile()

# --- Run the graph ---
if __name__ == "__main__":
    app = create_graph()
    
    print("Starting conversation...")    
    final = app.invoke({"name": "", "greeting": ""})    
    print("Final State:", final)
    print("Conversation complete!")
    
    # --- Draw and save the graph image ---
    print("Generating graph image...")
    graph = app.get_graph()
    graph_path = r"C:\code\agenticai_realpage\module_3\greeting_graph.png"
    graph.draw_mermaid_png(output_file_path=graph_path)
    print(f"Graph image saved at: {graph_path}")
