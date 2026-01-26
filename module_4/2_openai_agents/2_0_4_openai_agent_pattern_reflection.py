import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner
import json

load_dotenv(override=True)

# -------------------------------------------------
# REFLECTION AGENT (JSON OUTPUT)
# -------------------------------------------------
reflection_agent = Agent(
    name="ReflectionAgentJSON",
    model="gpt-4o-mini",
    instructions="""
You are a reflection-based agent.

Follow this exact format STRICTLY.

Produce output as valid JSON with three keys:

{
  "Draft": "...",
  "Reflection": "...",
  "RevisedAnswer": "..."
}

Draft:
- Produce an initial answer to the user query.

Reflection:
- Critically evaluate the Draft.
- Identify missing details, inaccuracies, or improvements.

RevisedAnswer:
- Improve the Draft using the Reflection.
- This is the final answer shown to the user.

Do NOT skip any section.
Ensure the output is valid JSON, parsable by a program.
"""
)

# -------------------------------------------------
# RUNNER LOOP
# -------------------------------------------------
async def main():
    # Query 1
    result1 = await Runner.run(
        reflection_agent,
        """
        Draft a customer-facing explanation for an outage where
        background jobs were delayed for 2 hours due to a database
        connection pool misconfiguration.

        Audience:
        - Non-technical customers
        - Paying enterprise users

        Constraints:
        - Do not blame individuals
        - Avoid internal jargon
        - Must include next steps and prevention measures
        """
    )

    print("\n===== RAW OUTPUT =====\n")
    print(result1.final_output)

    # Parse JSON safely
    try:
        data1 = json.loads(result1.final_output)
        print("\n===== PARSED REFLECTION =====\n")
        print("Draft:\n", data1.get("Draft", "N/A"))
        print("\nReflection:\n", data1.get("Reflection", "N/A"))
        print("\nRevised Answer:\n", data1.get("RevisedAnswer", "N/A"))
    except json.JSONDecodeError:
        print("\nERROR: Could not parse JSON output. Check agent formatting.")

    # Query 2 (product-grade, different example)
    result2 = await Runner.run(
        reflection_agent,
        "Explain how an AI agent should handle ambiguous user requests in a production system."
    )
    print("\n===== RAW OUTPUT =====\n")
    print(result2.final_output)

    # Parse JSON safely
    try:
        data2 = json.loads(result2.final_output)
        print("\n===== PARSED REFLECTION =====\n")
        print("Draft:\n", data2.get("Draft", "N/A"))
        print("\nReflection:\n", data2.get("Reflection", "N/A"))
        print("\nRevised Answer:\n", data2.get("RevisedAnswer", "N/A"))
    except json.JSONDecodeError:
        print("\nERROR: Could not parse JSON output. Check agent formatting.")

if __name__ == "__main__":
    asyncio.run(main())
