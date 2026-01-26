import asyncio
from agents import Agent, Runner, function_tool
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

# --------------------------------
# Tool call
# --------------------------------
@function_tool
def get_temperature(city: str) -> str:
    """Returns mock temperature for a city."""
    temps = {
        "new york": "72°F",
        "london": "65°F",
        "tokyo": "78°F",
        "sydney": "68°F",
        "mumbai": "85°F"
    }
    return temps.get(city.lower(), "70°F")

@function_tool
def get_time(timezone: str) -> str:
    """Returns current time for a timezone."""
    return datetime.now().strftime("%I:%M %p")

@function_tool
def calculate(expression: str) -> str:
    """Evaluates a simple math expression."""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except:
        return "Invalid expression"

# --------------------------------
# SIMPLE AGENT
# --------------------------------
agent = Agent(
    name="QuickAssistant",
    model="gpt-4o-mini",
    tools=[get_temperature, get_time, calculate],
    instructions="You are a helpful assistant. Use tools when needed. Be concise."
)

# --------------------------------
# MAIN
# --------------------------------
async def main():
    queries = [
        "What's the temperature in Tokyo?",
        "Calculate 25 * 4 + 10",
        "What time is it in New York timezone?",
    ]
    
    print("Processing queries in parallel...\n")
    
    # Process all queries concurrently
    tasks = [Runner.run(agent, query) for query in queries]
    results = await asyncio.gather(*tasks)
    
    # Display results
    for query, result in zip(queries, results):
        print(f"Q: {query}")
        print(f"A: {result.final_output}\n")

if __name__ == "__main__":
    asyncio.run(main())