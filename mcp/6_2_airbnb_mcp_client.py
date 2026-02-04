import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import datetime, timedelta

async def main():
    server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            today = datetime.now()
            checkin = today + timedelta(days=3)
            checkout = today + timedelta(days=7)
            
            # Get listings
            result = await session.call_tool("airbnb_search", {
                "location": "London",
                "checkin": checkin,
                "checkout": checkout,
                "adults": 2,
                "children": 2,
            })
            print(result.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())