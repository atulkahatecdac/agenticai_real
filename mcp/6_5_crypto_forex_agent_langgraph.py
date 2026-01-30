import asyncio
import re
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.graph import StateGraph, END
from typing import TypedDict

load_dotenv(override=True)

# Global MCP sessions
crypto_session = None
exchange_session = None

# ---------------- State ---------------- #
class State(TypedDict):
    crypto: str
    usd_price: float
    usd_result: str
    inr_result: str

# ---------------- Node 1: Get Crypto Price in USD ---------------- #
async def get_crypto_node(state: State) -> State:
    """Node that calls MCP server to get crypto price in USD"""
    global crypto_session
    result = await crypto_session.call_tool(
        "get_cryptocurrency_price", 
        {"crypto": state["crypto"]}
    )
    state["usd_result"] = result.content[0].text
    
    # Extract USD price from result
    match = re.search(r'\$([0-9,]+\.?[0-9]*)', state["usd_result"])
    if match:
        state["usd_price"] = float(match.group(1).replace(',', ''))
    
    return state

# ---------------- Node 2: Convert USD to INR ---------------- #
async def convert_to_inr_node(state: State) -> State:
    """Node that calls MCP server to convert USD to INR"""
    global exchange_session
    result = await exchange_session.call_tool(
        "convert_currency",
        {
            "from_currency": "USD",
            "to_currency": "INR",
            "amount": state["usd_price"]
        }
    )
    state["inr_result"] = result.content[0].text
    return state

# ---------------- Build Graph ---------------- #
workflow = StateGraph(State)
workflow.add_node("get_crypto", get_crypto_node)
workflow.add_node("convert_to_inr", convert_to_inr_node)
workflow.set_entry_point("get_crypto")
workflow.add_edge("get_crypto", "convert_to_inr")
workflow.add_edge("convert_to_inr", END)
graph = workflow.compile()

# ---------------- Main ---------------- #
async def main():
    global crypto_session, exchange_session
    
    # Initialize Crypto MCP connection
    crypto_params = StdioServerParameters(
        command="python", 
        args=["c:/code/agenticai/6_mcp/6_3_crypto_mcp_server.py"]
    )
    
    # Initialize Exchange Rate MCP connection
    exchange_params = StdioServerParameters(
        command="python",
        args=["c:/code/agenticai/6_mcp/6_5_forex_mcp_server.py"]
    )
    
    async with stdio_client(crypto_params) as (crypto_read, crypto_write):
        async with ClientSession(crypto_read, crypto_write) as c_session:
            crypto_session = c_session
            await c_session.initialize()
            
            async with stdio_client(exchange_params) as (exchange_read, exchange_write):
                async with ClientSession(exchange_read, exchange_write) as e_session:
                    exchange_session = e_session
                    await e_session.initialize()
                    
                    # Run the graph
                    result = await graph.ainvoke({
                        "crypto": "bitcoin",
                        "usd_price": 0.0,
                        "usd_result": "",
                        "inr_result": ""
                    })
                    
                    print("=== Crypto to INR Workflow ===")
                    print(f"Step 1 - USD Price: {result['usd_result']}")
                    print(f"Step 2 - INR Conversion: {result['inr_result']}")

if __name__ == "__main__":
    asyncio.run(main())