import asyncio
import json
from dotenv import load_dotenv
from agents import Agent, Runner

load_dotenv(override=True)

# --------------------------------
# REWOO AGENT WITH AGENTIC CONSTRAINTS
# --------------------------------
admission_agent = Agent(
    name="StudentAdmissionAgenticREWOO",
    model="gpt-4o-mini",
    instructions="""
You are a REWOO agent for a Student Admission System.

Use ONLY the ontology below and strictly follow the PROCESS.

OUTPUT MUST BE valid JSON with keys:
{
  "Applicant": {"name": "...", "marks": ..., "interview_score": ..., "statement": "..."},
  "Program": {"name": "...", "cutoff_marks": ...},
  "Eligibility": {"status": "...", "reason": "..."},
  "Decision": "...",
  "Justification": "..."
}

--------------------
ONTOLOGY
--------------------
Entities:
- Applicant: {name, marks, interview_score, statement}
- Program: {name, cutoff_marks}
- Eligibility: {status, reason}
- Decision: {admit | reject}

Constraints:
- Numeric Eligibility: (marks + interview_score) / 2 >= cutoff → eligible
- Essay Consideration: if applicant slightly below cutoff (<5% below), essay may justify admit
- Otherwise: not eligible

Valid Operations:
- Decompose(Task)
- Evaluate(Eligibility)
- Decide(Decision)
- Reflect on trade-offs between numeric score and statement quality

--------------------
PROCESS
--------------------
1. Reason: Identify applicant, program, constraints
2. Evaluate: Apply numeric + essay-based rules
3. Work: Perform decision agentically
4. Output: Return structured JSON with justification
"""
)

# --------------------------------
# PROCESS EACH STUDENT
# --------------------------------
async def process_student(student):
    query = f"""
Applicant Details:
- Name: {student['name']}
- Marks: {student['marks']}
- Interview Score: {student['interview_score']}
- Statement: {student['statement']}

Program:
- Name: {student['program']['name']}
- Cutoff Marks: {student['program']['cutoff_marks']}

Determine admission decision considering both numeric scores and statement strength.
"""
    result = await Runner.run(admission_agent, query)

    # Safe JSON parsing
    try:
        data = json.loads(result.final_output)
        print(f"\n=== Student: {student['name']} ===")
        print("Decision:", data.get("Decision"))
        print("Eligibility Reason:", data.get("Eligibility", {}).get("reason"))
        print("Justification:", data.get("Justification"))
    except json.JSONDecodeError:
        print(f"\nERROR parsing JSON for {student['name']}")
        print(result.final_output)

# --------------------------------
# MAIN LOOP
# --------------------------------
async def main():
    with open(r"c:\code\agenticai\2_openai_agents\students.json", "r") as f:
        students = json.load(f)

    for student in students:
        await process_student(student)

if __name__ == "__main__":
    asyncio.run(main())
