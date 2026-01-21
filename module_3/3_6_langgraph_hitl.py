# pip install langgraph

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

# ======================================================
# 1. STATE (Schema = Memory)
# ======================================================

class TicketState(TypedDict):
    ticket_text: str

    category: Literal["billing", "technical", "account", "unknown"]
    draft_reply: str
    confidence: float

    status: Literal[
        "new",
        "auto_approved",
        "needs_human",
        "approved",
        "rejected"
    ]

    human_feedback: str
    iterations: int


# ======================================================
# 2. NODES
# ======================================================

def analyze_ticket(state: TicketState) -> TicketState:
    """
    Automated LLM-style ticket triage
    """
    text = state["ticket_text"].lower()

    if "refund" in text:
        category = "billing"
        reply = "Your refund request has been received and is being processed."
        confidence = 0.9
    elif "error" in text or "not working" in text:
        category = "technical"
        reply = "Please try restarting the application and let us know if the issue persists."
        confidence = 0.6
    else:
        category = "unknown"
        reply = "Thank you for contacting support. We are reviewing your request."
        confidence = 0.4

    return {
        "category": category,
        "draft_reply": reply,
        "confidence": confidence,
        "iterations": state["iterations"] + 1
    }


def auto_send(state: TicketState) -> TicketState:
    """
    Auto-approved response
    """
    print("\nAUTO-SENT RESPONSE:")
    print(state["draft_reply"])
    return {"status": "auto_approved"}


def human_review(state: TicketState) -> TicketState:
    """
    REAL Human-in-the-loop approval
    """
    print("\nHUMAN REVIEW REQUIRED")
    print("Ticket:", state["ticket_text"])
    print("Proposed reply:", state["draft_reply"])
    print(f"Confidence: {state['confidence']}")

    decision = input("\nApprove reply? (yes / no): ").strip().lower()

    if decision == "yes":
        notes = input("Optional feedback / notes: ")
        return {
            "status": "approved",
            "human_feedback": notes
        }
    else:
        reason = input("Reason for rejection: ")
        return {
            "status": "rejected",
            "human_feedback": reason
        }


def reject_ticket(state: TicketState) -> TicketState:
    print("\nTICKET REJECTED")
    print("Reason:", state["human_feedback"])
    return {"status": "rejected"}


# ======================================================
# 3. MEMORY-AWARE ROUTING
# ======================================================

def route_after_analysis(state: TicketState) -> str:
    if state["confidence"] >= 0.8:
        return "auto_send"
    return "human_review"


def route_after_human(state: TicketState) -> str:
    return state["status"]  # approved | rejected


# ======================================================
# 4. BUILD GRAPH
# ======================================================

builder = StateGraph(TicketState)

builder.add_node("analyze", analyze_ticket)
builder.add_node("auto_send", auto_send)
builder.add_node("human_review", human_review)
builder.add_node("reject", reject_ticket)

builder.set_entry_point("analyze")

builder.add_conditional_edges(
    "analyze",
    route_after_analysis,
    {
        "auto_send": "auto_send",
        "human_review": "human_review"
    }
)

builder.add_conditional_edges(
    "human_review",
    route_after_human,
    {
        "approved": "auto_send",
        "rejected": "reject"
    }
)

builder.add_edge("auto_send", END)
builder.add_edge("reject", END)

graph = builder.compile()

# ======================================================
# 5 PRINT / SAVE GRAPH IMAGE
# ======================================================

graph_image_bytes = graph.get_graph().draw_mermaid_png()

with open(r"c:\code\agenticai_realpage\module_3\support_ticket_langgraph.png", "wb") as f:
    f.write(graph_image_bytes)

print("\nGraph image saved as: support_ticket_langgraph.png")

# ======================================================
# 6. RUN WORKFLOW
# ======================================================

result = graph.invoke({
    "ticket_text": "The app is not working after update",
    "category": "unknown",
    "draft_reply": "",
    "confidence": 0.0,
    "status": "new",
    "human_feedback": "",
    "iterations": 0
})

print("\nFINAL STATE:")
print(result)
