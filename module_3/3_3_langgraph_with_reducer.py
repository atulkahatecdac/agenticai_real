from typing import TypedDict, List
from typing_extensions import Annotated
from langgraph.graph import StateGraph, END

# -------------------
# Reducer
# -------------------
def add_messages(old: List[str], new: List[str]) -> List[str]:
    return old + new

# -------------------
# State (WITH reducer)
# -------------------
class State(TypedDict):
    messages: Annotated[List[str], add_messages]

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
