import asyncio
import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load .env for key and profile
load_dotenv(override=True)

SMITHERY_KEY = os.getenv("SMITHERY_API_KEY")
SMITHERY_PROFILE = os.getenv("SMITHERY_PROFILE", "default-profile")

if not SMITHERY_KEY:
    raise ValueError("Missing SMITHERY_API_KEY. Please set it in your .env file.")

async def main():
    # Define the GitHub MCP server via Smithery
    server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@smithery/cli@latest",
            "run",
            "@smithery-ai/github",
            "--key",
            SMITHERY_KEY,
            "--profile",
            SMITHERY_PROFILE,
        ],
    )

    # Connect and start session
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List all available tools
            tools_response = await session.list_tools()
            print("\n=== Available GitHub MCP Tools ===\n")
            for t in tools_response.tools:
                print(f"{t.name} — {t.description}")
            
            print(f"\nTotal tools found: {len(tools_response.tools)}")

if __name__ == "__main__":
    asyncio.run(main())
