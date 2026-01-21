from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# -------------------
# State (NO reducers)
# -------------------
class State(TypedDict):
    messages: List[str]

# -------------------
# Nodes
# -------------------
def agent_a(state: State):
    return {"messages": ["Hello from Agent A"]}

def agent_b(state: State):
    return {"messages": ["Hello from Agent B"]}

# -------------------
# Graph
# -------------------
graph = StateGraph(State)

graph.add_node("agent_a", agent_a)
graph.add_node("agent_b", agent_b)

graph.set_entry_point("agent_a")
graph.add_edge("agent_a", "agent_b")
graph.add_edge("agent_b", END)

app = graph.compile()

# -------------------
# Run
# -------------------
result = app.invoke({"messages": []})
print(result)
