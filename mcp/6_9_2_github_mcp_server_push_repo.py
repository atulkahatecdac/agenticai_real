import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load Smithery API key securely from .env or environment variable
from dotenv import load_dotenv
load_dotenv(override=True)

SMITHERY_API_KEY = os.getenv("SMITHERY_API_KEY")

if not SMITHERY_API_KEY:
    raise ValueError("Missing SMITHERY_API_KEY in environment variables or .env file")

async def main():
    server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@smithery/cli@latest",
            "run",
            "@smithery-ai/github",
            "--key",
            SMITHERY_API_KEY,
        ],
    )

    file_path = r"C:\agenticainew\2_Openai_agents\2_2_openai_agent_tool_hardcoded_knowledge.py"
    repo_name = "agenticainew"
    owner = "newdelthis"
    branch = "main"

    # Read the file content
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    payload = {
        "repo": repo_name,
        "owner": owner,
        "branch": branch,
        "message": "Pushing using GitHub MCP Server (single file)",
        "files": [
            {"path": "2_Openai_agents/2_2_openai_agent_tool_hardcoded_knowledge.py", "content": content}
        ],
    }

    print(f"Preparing to push: {file_path}")
    print("Connecting to GitHub MCP server...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected. Calling 'push_files' tool...")

            try:
                result = await asyncio.wait_for(session.call_tool("push_files", payload), timeout=90)
                print("Push completed successfully!")
                print(result.content[0].text)
            except asyncio.TimeoutError:
                print("Timed out while pushing to GitHub.")
            except Exception as e:
                print(f"Error during push: {e}")

if __name__ == "__main__":
    asyncio.run(main())
