import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Create server parameters
    server_params = StdioServerParameters(
        command="python", 
        args=["c:/code/agenticai/6_mcp/6_6_indian_jobs_mcp_server.py"]  # Use forward slashes
    )
    
    # Use stdio_client to get read/write streams
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print("\nAvailable MCP Tools:")
            for t in tools.tools:
                print(f" - {t.name}")
            
            # Call the tool search_jobs
            print("\nCalling search_jobs tool...\n")

            result = await session.call_tool(
                name="search_jobs",
                arguments={
                    "limit": "5",  # Use int, not string
                    "location": "Pune",
                    "title": "AI"
                    # Don't need to pass None values
                }
            )

            print("Response from server:")
            print(result.content[0].text)

# Run the async client
if __name__ == "__main__":
    asyncio.run(main())