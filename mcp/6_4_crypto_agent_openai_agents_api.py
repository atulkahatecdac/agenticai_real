import asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agents import Agent, Runner, function_tool

load_dotenv(override=True)

# Global MCP session
mcp_session = None
mcp_read = None
mcp_write = None

# ---------------- MCP Tool ---------------- #
@function_tool
async def get_crypto_price(crypto: str) -> str:
    """Get cryptocurrency price in USD
    
    Args:
        crypto: The cryptocurrency name (e.g., 'bitcoin', 'ethereum')
    
    Returns:
        Price information as a string
    """
    global mcp_session
    result = await mcp_session.call_tool(
        "get_cryptocurrency_price", {"crypto": crypto}
    )
    return result.content[0].text

# ---------------- Crypto Agent ---------------- #
crypto_agent = Agent(
    name="Crypto_Price_Agent",
    instructions="Use the get_crypto_price tool to fetch cryptocurrency prices when asked.",
    tools=[get_crypto_price]
)

# ---------------- Main ---------------- #
async def main():
    global mcp_session, mcp_read, mcp_write
    
    # Initialize MCP connection once
    server_params = StdioServerParameters(
        command="python", 
        args=["c:/code/agenticai/6_mcp/6_3_crypto_mcp_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        mcp_read, mcp_write = read, write
        async with ClientSession(read, write) as session:
            mcp_session = session
            await session.initialize()
            
            # Now run the agent
            result = await Runner.run(
                crypto_agent, 
                "Get the price of dogecoin in USD."
            )
            print("=== Crypto Agent ===")
            print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())