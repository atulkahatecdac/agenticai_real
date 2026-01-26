from dotenv import load_dotenv
from agents import Agent, Runner

load_dotenv(override=True)

instruction = "You are a helpful customer support analyst."
message = ("Analyze this customer feedback and suggest improvements to the product: "
           # "'The packaging is great, but the delivery was delayed by two days.'")
           "'The product was a bit stale'")

agent = Agent(
    name="Assistant",
    instructions=instruction,
    model="gpt-4o-mini", # It is optional - it defaults to gpt-4o-mini
)

# Runner.run initializes a session object to keep a track of the
#  agent's actions, memory, and output
# Takes the user's input (message) and sends it to the LLM
#  Internally, it will call the Chat Completions/Responses API
# Gets back a response from the LLM as result, which is a Session object
result = Runner.run_sync(agent, message)
print(result.final_output)
