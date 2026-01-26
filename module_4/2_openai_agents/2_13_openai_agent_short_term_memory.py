# pip install python-dotenv openai-agents

import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, SQLiteSession, trace

load_dotenv()

nutrition_agent = Agent(
    name="Nutrition Assistant",
    instructions="""
    You are a helpful assistant comparing how healthy different foods are.
    If you answer, give a list of how healthy the foods are with a score from 1 to 10.
    Order by: healthiest food comes first.

    Example:
    Q: Compare X and Y
    A:
    1) X: 8/10 - Very healthy but high in fructose
    2) Y: 3/10 - High in sugar and fat
    """,
)

async def main():
    # ---- No memory ----
    print("\n********************")
    print("\n=== No Memory ===")
    print("\n********************")
    result = await Runner.run(
        nutrition_agent, "Which is healthier, bananas or lollipop?"
    )
    print(result.final_output)

    result = await Runner.run(
        nutrition_agent, "Add apples to the comparison"
    )
    print(result.final_output)

    # ---- With short-term memory ----
    print("\n********************")
    print("\n=== Short Term Memory ===")
    print("\n********************")
    session = SQLiteSession("conversation_history")

    result = await Runner.run(
        nutrition_agent,
        "Which is healthier, bananas or lollipop?",
        session=session,
    )
    print(result.final_output)

    with trace("Simple Nutrition Agent"):
        result = await Runner.run(
            nutrition_agent,
            "Add apples to the comparison",
            session=session,
        )
        print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
