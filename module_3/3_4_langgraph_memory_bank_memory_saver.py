from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Define the structure of data that will be tracked across all transactions
# This state persists in memory between function calls
class BankingState(TypedDict):
    # Dictionary mapping account names to their balances
    accounts: dict
    # Counter tracking how many transactions have been processed
    tx_count: int
    # These next three fields are temporary inputs for each transaction
    # They are passed in with each invoke call but not saved long term
    acc: str
    op: str
    amt: float

# This function processes a single banking transaction
# It receives the current state, modifies it, and returns the updates
def banker(state: BankingState):
    # Extract the transaction details from the input state
    acc = state["acc"]
    op = state["op"]
    amt = state["amt"]
    
    # Get a copy of the current account balances dictionary
    # Using copy prevents accidentally modifying the original state
    accounts = state.get("accounts", {}).copy()
    # Look up the current balance for this account, default to 0 if new
    current = accounts.get(acc, 0)
    
    # Calculate the new balance based on the operation type
    if op == "credit":
        accounts[acc] = current + amt
    else:
        # Debit operation subtracts from the balance
        accounts[acc] = current - amt
    
    # Return the updates to be merged into the persistent state
    # Only these returned fields will be saved
    return {
        "accounts": accounts,
        "tx_count": state.get("tx_count", 0) + 1
    }

builder = StateGraph(BankingState)

builder.add_node("banker", banker)
builder.set_entry_point("banker")

builder.add_edge("banker", END)

app = builder.compile(checkpointer=MemorySaver())

# Main loop that handles user interaction and transaction processing
def run():
    # Configuration object that identifies this specific conversation thread
    # All state is stored under this thread ID
    config = {"configurable": {"thread_id": "77"}}
    
    # Infinite loop to keep accepting transactions
    while True:
        # Retrieve the current saved state from memory
        state = app.get_state(config).values
        # Get the transaction count, defaulting to 0 if this is the first run
        count = state.get("tx_count", 0)
        
        # Simulate a system crash after 5 transactions
        if count >= 5:
            print("!!! CRASH !!!")
            # Force exit the program
            raise SystemExit
        
        # Prompt user for account number, show current position out of 5
        acc = input(f"[{count+1}/5] Account: ")
        # Allow user to exit gracefully by entering 0
        if acc == "0":
            break
        
        # Get the operation type credit or debit    
        op = input("Op (credit/debit): ")
        # Get the transaction amount as a number
        amt = float(input("Amount: "))
        
        # Send the transaction data to the graph for processing
        # This will call the banker function and save the results
        app.invoke({"acc": acc, "op": op, "amt": amt}, config)
        
        # Retrieve the updated state after the transaction was processed
        new_state = app.get_state(config).values

        # Display the new balance for the account that was just modified
        print(f"Balance for {acc}: ${new_state['accounts'].get(acc, 0)}")

if __name__ == "__main__":
    try:
        run()
    except SystemExit:
        print("\nTry running again...")
