import asyncio
from agents import Agent, Runner, function_tool, trace
from dotenv import load_dotenv

load_dotenv()

# Mock customer database
CUSTOMER_DB = {
    "C001": {
        "name": "Alice Johnson",
        "credit_score": 720,
        "active_loans": [
            {"type": "home", "emi": 25000},
            {"type": "personal", "emi": 8000},
        ],
        "monthly_income": 90000,
    },
    "C002": {
        "name": "Bob Smith",
        "credit_score": 610,
        "active_loans": [
            {"type": "personal", "emi": 12000},
        ],
        "monthly_income": 45000,
    },
}

# Customer profile agent
@function_tool
def get_customer_profile(customer_id: str) -> dict:
    """Fetch customer loan and income details."""
    profile = CUSTOMER_DB.get(customer_id, {})
    print(f"[TOOL] Fetching profile for {customer_id}: {profile.get('name', 'Not found')}")
    return profile

customer_profile_agent = Agent(
    name="Customer Profile Agent",
    instructions="You fetch customer financial profile. Return structured information only.",
    tools=[get_customer_profile],
)

# Risk evaluation agent
@function_tool
def evaluate_risk(customer_id: str) -> str:
    """
    Simple risk logic:
    - EMI ratio > 40% = High Risk
    - Credit score < 650 = High Risk
    """
    profile = CUSTOMER_DB.get(customer_id, {})
    
    if not profile:
        return "Customer not found"

    total_emi = sum(loan["emi"] for loan in profile["active_loans"])
    emi_ratio = total_emi / profile["monthly_income"]
    
    print(f"[TOOL] Risk eval for {customer_id}: EMI ratio={emi_ratio:.2%}, Credit={profile['credit_score']}")

    if emi_ratio > 0.4 or profile["credit_score"] < 650:
        return f"High Risk (EMI ratio: {emi_ratio:.2%}, Credit: {profile['credit_score']})"
    return f"Low Risk (EMI ratio: {emi_ratio:.2%}, Credit: {profile['credit_score']})"

risk_agent = Agent(
    name="Risk Evaluation Agent",
    instructions="You evaluate loan risk based on EMI ratio and credit score. Return only risk category.",
    tools=[evaluate_risk],
)

# Convert agents into tools
customer_profile_tool = customer_profile_agent.as_tool(
    tool_name="customer-profile",
    tool_description="Fetch customer loan and income details",
)

risk_evaluation_tool = risk_agent.as_tool(
    tool_name="risk-evaluator",
    tool_description="Evaluate loan risk category",
)

# Loan decision agent (Orchestrator)
loan_decision_agent = Agent(
    name="Loan Decision Agent",
    instructions="""
    You are a banking loan decision assistant.

    Workflow:
    1) Use customer-profile tool to get customer details
    2) Use risk-evaluator tool to assess risk
    3) Make loan decision based on risk

    Decision Rules:
    - Low Risk → Recommend approval
    - High Risk → Recommend rejection or reduced amount

    Always explain reasoning clearly with specific numbers.
    """,
    tools=[customer_profile_tool, risk_evaluation_tool],
)

# Test
async def main():
    print("=== Testing Customer C001 ===")
    with trace("Loan Eligibility C001"):
        result = await Runner.run(
            loan_decision_agent,
            "Check loan eligibility for customer C001"
        )
        print("\n" + result.final_output + "\n")
    
    print("\n=== Testing Customer C002 ===")
    with trace("Loan Eligibility C002"):
        result = await Runner.run(
            loan_decision_agent,
            "Check loan eligibility for customer C002"
        )
        print("\n" + result.final_output + "\n")


if __name__ == "__main__":
    asyncio.run(main())