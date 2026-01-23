# Same code as before, but we now persist the memory on the disk using SqliteSaver
# Note that this will create a bank.db file on the disk

import sqlite3
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# State: What persists between transactions
class BankingState(TypedDict):
    accounts: dict
    tx_count: int
    # Input fields (not persisted, just passed in)
    acc: str
    op: str
    amt: float

# Logic: Process one transaction
def banker(state: BankingState):
    acc = state["acc"]
    op = state["op"]
    amt = state["amt"]
    
    accounts = state.get("accounts", {}).copy()
    current = accounts.get(acc, 0)
    
    if op == "credit":
        accounts[acc] = current + amt
    else:
        accounts[acc] = current - amt
    
    return {
        "accounts": accounts,
        "tx_count": state.get("tx_count", 0) + 1
    }

# Setup graph with SQLite
builder = StateGraph(BankingState)
builder.add_node("banker", banker)
builder.set_entry_point("banker")
builder.add_edge("banker", END)

# check_same_thread=False
# Allows the same connection to be used across multiple threads
# We  are taking responsibility for thread safety ourself
conn = sqlite3.connect(r"c:\code\agenticai\3_langgraph\bank.db", check_same_thread=False)
app = builder.compile(checkpointer=SqliteSaver(conn))

# Run banking loop
def run():
    config = {"configurable": {"thread_id": "77"}}
    
    # Check for crash recovery on startup
    state = app.get_state(config).values
    if state and state.get("tx_count", 0) >= 5:
        print(f"RECOVERED from crash! Accounts: {state.get('accounts', {})}")
        app.update_state(config, {"tx_count": 0})
    
    while True:
        state = app.get_state(config).values
        count = state.get("tx_count", 0)
        
        if count >= 5:
            print("!!! CRASH !!!")
            raise SystemExit
        
        acc = input(f"[{count+1}/5] Account: ")
        if acc == "0":
            break
            
        op = input("Op (credit/debit): ")
        amt = float(input("Amount: "))
        
        app.invoke({"acc": acc, "op": op, "amt": amt}, config)
        
        # Show updated balance
        new_state = app.get_state(config).values
        print(f"Balance for {acc}: ${new_state['accounts'].get(acc, 0)}")

if __name__ == "__main__":
    run()