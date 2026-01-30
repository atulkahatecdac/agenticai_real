import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def discover_tools():
    """Connect to MCP server and discover available tools"""
    
    # Configure the server connection
    server_params = StdioServerParameters(
        command="python",
        args=[r"C:\code\agenticai\6_mcp\6_0_database_mcp_server.py"],  # Replace with your server script name
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            # List all available tools
            tools_list = await session.list_tools()
            
            print("Available MCP Tools:")
            print("=" * 60)
            
            for tool in tools_list.tools:
                print(f"\nTool Name: {tool.name}")
                print(f"Description: {tool.description}")
                
                if tool.inputSchema:
                    print("Parameters:")
                    properties = tool.inputSchema.get("properties", {})
                    required = tool.inputSchema.get("required", [])
                    
                    for param_name, param_info in properties.items():
                        req_marker = " (required)" if param_name in required else " (optional)"
                        param_type = param_info.get("type", "unknown")
                        param_desc = param_info.get("description", "No description")
                        print(f"  - {param_name} ({param_type}){req_marker}: {param_desc}")
                
                print("-" * 60)
            
            print(f"\nTotal tools found: {len(tools_list.tools)}")

if __name__ == "__main__":
    asyncio.run(discover_tools())