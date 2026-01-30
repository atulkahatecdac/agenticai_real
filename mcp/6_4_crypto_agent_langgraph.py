from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.graph import StateGraph, END
from typing import TypedDict
import asyncio

load_dotenv(override=True)

# Global MCP session
mcp_session = None

# ---------------- State ---------------- #
class State(TypedDict):
    crypto: str
    result: str

# ---------------- Node ---------------- #
async def get_crypto_node(state: State) -> State:
    """Node that calls MCP server to get crypto price"""
    global mcp_session
    result = await mcp_session.call_tool(
        "get_cryptocurrency_price", 
        {"crypto": state["crypto"]}
    )
    state["result"] = result.content[0].text
    return state

# ---------------- Build Graph ---------------- #
workflow = StateGraph(State)
workflow.add_node("get_crypto", get_crypto_node)
workflow.set_entry_point("get_crypto")
workflow.add_edge("get_crypto", END)
graph = workflow.compile()

# ---------------- Main ---------------- #
async def main():
    global mcp_session
    
    # Initialize MCP connection
    server_params = StdioServerParameters(
        command="python", 
        args=["c:/code/agenticai/6_mcp/6_3_crypto_mcp_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            mcp_session = session
            await session.initialize()
            
            # Run the graph
            result = await graph.ainvoke({
                "crypto": "bitcoin",
                "result": ""
            })
            
            print("=== Crypto LangGraph ===")
            print(result["result"])

if __name__ == "__main__":
    asyncio.run(main())