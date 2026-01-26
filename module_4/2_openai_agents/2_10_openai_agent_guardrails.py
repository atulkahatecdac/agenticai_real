# pip install pydantic python-dotenv openai-agents

import asyncio
from pydantic import BaseModel
from agents import Agent, GuardrailFunctionOutput, InputGuardrailTripwireTriggered, RunContextWrapper, Runner, input_guardrail
from dotenv import load_dotenv

load_dotenv(override=True)

ACCOUNTS = [
    {"account_number": "A001", "name": "Alice", "balance": 1500, "type": "Savings", "branch": "West Wing"},
    {"account_number": "A002", "name": "Bob", "balance": 7800, "type": "Current", "branch": "North Street"},
    {"account_number": "A003", "name": "Charlie", "balance": 3200, "type": "Savings", "branch": "East Street"},
]

class BalanceQueryOutput(BaseModel):
    is_balance_query: bool
    reasoning: str

guardrail_agent = Agent(
    name="Balance Guardrail",
    model="gpt-4o-mini",
    instructions="Block balance queries (balance, money, funds, amount). Allow account type and branch queries.",
    output_type=BalanceQueryOutput,
)

@input_guardrail
async def balance_guardrail(ctx: RunContextWrapper[None], agent: Agent, input):
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_balance_query,
    )

agent = Agent(
    name="Banking Support",
    model="gpt-4o-mini",
    instructions=f"""Banking support agent. Never reveal balances. 
    
Account database:
{ACCOUNTS}

You can share: account type, branch name, account holder name.
You cannot share: balance amounts.""",
    input_guardrails=[balance_guardrail],
)

async def main():
    # Test 1: Block balance query
    try:
        await Runner.run(agent, "What is the balance for A001?")
        print("Guardrail failed")
    except InputGuardrailTripwireTriggered:
        print("Balance query blocked")

    # Test 2: Allow account type query
    output = await Runner.run(agent, "What type of account is A002?")
    print(f"Response: {output.final_output}")

if __name__ == "__main__":
    print("Checking for the following queries:")
    print("What is the balance for A001?")
    print("What type of account is A002?")
    asyncio.run(main())