# Two tasks running in sychronous mode

import time
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

start = time.time()

# Run them sequentially (synchronous execution)
print("Running UPI Essay agent...")
t1 = time.time()
result1 = Runner.run_sync(upi_agent, "Write the essay now.")
print(f"UPI Essay completed in {time.time() - t1:.2f} seconds\n")

print("Running Agent API Note agent...")
t2 = time.time()
result2 = Runner.run_sync(agent_api_agent, "Write the note now.")
print(f"Agent API Note completed in {time.time() - t2:.2f} seconds\n")

print("=== Essay on UPI ===\n")
print(result1.final_output)

print("\n=== Note on Agent API ===\n")
print(result2.final_output)

print(f"\nTotal execution time: {time.time() - start:.2f} seconds")
