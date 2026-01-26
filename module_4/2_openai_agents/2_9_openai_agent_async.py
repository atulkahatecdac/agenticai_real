# Two agents in async mode
import time
import asyncio
from agents import Agent, Runner
from dotenv import load_dotenv

load_dotenv(override=True)

# Define two agents with different instructions
upi_agent = Agent(
    name="UPI_Essay_Writer",
    instructions="Write a clear, well-structured essay in 4 paragraphs about UPI."
)

agent_api_agent = Agent(
    name="AgentAPI_Note_Writer",
    instructions="Write a detailed one-page note explaining the OpenAI Agents API: what it is, how it works, and its use cases."
)

async def main():
    start = time.time()

    print("Running agents concurrently...\n")

    # Run both agents concurrently
    t1 = time.time()
    task1 = Runner.run(upi_agent, "Write the essay now.")
    task2 = Runner.run(agent_api_agent, "Write the note now.")

    result1, result2 = await asyncio.gather(task1, task2)

    print(f"Both tasks completed in {time.time() - t1:.2f} seconds\n")

    print("=== Essay on UPI  ===\n")
    print(result1.final_output)

    print("\n=== Note on Agent API ===\n")
    print(result2.final_output)

    print(f"\nTotal execution time: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
