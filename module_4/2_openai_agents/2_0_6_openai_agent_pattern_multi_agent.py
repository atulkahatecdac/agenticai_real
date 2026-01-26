import asyncio
import json
import re
from dotenv import load_dotenv
from agents import Agent, Runner
from pathlib import Path

load_dotenv(override=True)

# --------------------------------
# UTILITY: Clean JSON output from agent
# --------------------------------
def extract_json(text: str) -> str:
    """
    Extract the first {...} block from text.
    This handles cases where the agent adds extra commentary.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text  # fallback

def safe_parse_json(text: str) -> dict:
    """
    Extract and parse JSON safely, with fallback to raw text.
    """
    try:
        clean_text = extract_json(text)
        return json.loads(clean_text)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON", "raw_output": text}

# --------------------------------
# AGENT 1: PLANNER
# --------------------------------
planner_agent = Agent(
    name="HRPlannerAgent",
    model="gpt-4o-mini",
    instructions="""
You are a Planner Agent for HR candidate filtering.

Task:
- Extract candidate details: skills, experience, projects, statement
- Summarize candidate suitability for the role 'Agentic AI Developer/Architect'

VERY IMPORTANT:
- Output MUST be valid JSON only.
- Do NOT include any extra explanation or text.
- Escape newlines inside strings.

JSON FORMAT:
{
  "Candidate": {
    "name": "...",
    "skills": [...],
    "experience_years": ...,
    "projects": [...],
    "statement": "..."
  },
  "Role": "Agentic AI Developer/Architect",
  "EvaluationFormula": "..."
}
"""
)

# --------------------------------
# AGENT 2: EVALUATOR
# --------------------------------
evaluator_agent = Agent(
    name="HREvaluatorAgent",
    model="gpt-4o-mini",
    instructions="""
You are an Evaluation Agent for HR.

Input:
- Planner Agent JSON output

Task:
- Compare candidate skills, experience, and projects to role requirements
- Consider statement quality for agentic AI experience
- Output MUST indicate whether candidate is suitable

VERY IMPORTANT:
- Output MUST be valid JSON only.
- Do NOT include extra text or commentary.

JSON FORMAT:
{
  "Suitability": {"status": "...", "reason": "..."}
}
"""
)

# --------------------------------
# AGENT 3: DECIDER
# --------------------------------
decider_agent = Agent(
    name="HRDeciderAgent",
    model="gpt-4o-mini",
    instructions="""
You are a Decision Agent for HR candidate selection.

Input:
- Planner JSON output
- Evaluator JSON output

Task:
- Decide whether to invite candidate for interview or reject
- Provide final justification

VERY IMPORTANT:
- Output MUST be valid JSON only.
- Do NOT include extra explanation or commentary.

JSON FORMAT:
{
  "Candidate": {...},
  "Role": "Agentic AI Developer/Architect",
  "Suitability": {...},
  "Decision": "...",
  "Justification": "..."
}
"""
)

# --------------------------------
# PROCESS SINGLE CANDIDATE
# --------------------------------
async def process_candidate(candidate: dict) -> dict:
    planner_input = f"""
Candidate Details:
- Name: {candidate['name']}
- Skills: {', '.join(candidate['skills'])}
- Experience (years): {candidate['experience_years']}
- Projects: {', '.join(candidate['projects'])}
- Statement: {candidate['statement']}

Role: Agentic AI Developer/Architect
"""
    # Step 1: Planning
    plan_result = await Runner.run(planner_agent, planner_input)
    plan_json = safe_parse_json(plan_result.final_output)

    # Step 2: Evaluation
    eval_result = await Runner.run(evaluator_agent, json.dumps(plan_json))
    eval_json = safe_parse_json(eval_result.final_output)

    # Step 3: Decision
    decider_input = json.dumps({
        "PlannerOutput": plan_json,
        "EvaluatorOutput": eval_json
    })
    decision_result = await Runner.run(decider_agent, decider_input)
    decision_json = safe_parse_json(decision_result.final_output)

    return decision_json

# --------------------------------
# MAIN LOOP
# --------------------------------
async def main():
    # Load candidates from JSON file
    with open(r"c:\code\agenticai\2_openai_agents\candidates.json", "r", encoding="utf-8") as f:
        candidates = json.load(f)

    # Output file (full JSON array)
    output_file = Path(r"c:\code\agenticai\2_openai_agents\candidate_decisions.json")
    all_decisions = []

    for candidate in candidates:
        decision_json = await process_candidate(candidate)
        all_decisions.append(decision_json)
        print(f"Processed candidate: {candidate['name']}")

    # Write all decisions to a single JSON array
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(all_decisions, f, indent=2)

    print(f"\nAll candidate decisions written to {output_file.resolve()}")

if __name__ == "__main__":
    asyncio.run(main())
