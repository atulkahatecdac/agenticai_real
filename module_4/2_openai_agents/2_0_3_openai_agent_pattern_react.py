import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner

load_dotenv(override=True)

# -------------------------------------------------
# GENERIC 3-CYCLE REACT AGENT
# -------------------------------------------------
react_agent = Agent(
    name="ReAct3CycleAgent",
    model="gpt-4o-mini",
    instructions="""
You are a generic ReAct (Reason + Act) agent.

You MUST follow this exact structure and COMPLETE EXACTLY 3 cycles.

For each cycle:

Thought (Cycle N):
- Reason about the problem.

Action (Cycle N):
- Describe ONE logical action.

Observation (Cycle N):
- Infer the outcome.

After 3 cycles, produce:

Final:
- Root cause(s)
- Recommendations
"""
)

# -------------------------------------------------
# RUNNER
# -------------------------------------------------
async def main():
    problem_statement = """
    Our CI/CD pipeline started failing intermittently after enabling
    parallel test execution.

    Symptoms:
    - Random test failures
    - Database deadlocks during integration tests
    - Failures disappear when tests are run sequentially
    """

    result = await Runner.run(react_agent, problem_statement)
    print(result.final_output)
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())