# pip install langchain-openai langchain-core langgraph python-dotenv requests

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os, requests
from langgraph.graph import StateGraph, END
from typing import TypedDict

# --------------------------
# Shared State
# --------------------------
class AgentState(TypedDict):
    code_url: str
    code_content: str
    security_issues: str
    suggestions: str
    final_report: str


# --------------------------
# Setup
# --------------------------
load_dotenv(override=True)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


# --------------------------
# Utility Function
# --------------------------
def fetch_code(repo_url: str, path: str = "") -> str:
    """Recursively fetch Java code from a GitHub repo."""
    repo_path = repo_url.replace("https://github.com/", "")
    api_url = f"https://api.github.com/repos/{repo_path}/contents/{path}"
    code_files = []

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        items = response.json()

        for item in items:
            if item["type"] == "file" and item["name"].endswith(".java"):
                file_content = requests.get(item["download_url"]).text
                code_files.append(f"// {item['path']}\n{file_content}\n")
            elif item["type"] == "dir":
                sub_code = fetch_code(repo_url, item["path"])
                code_files.append(sub_code)

        code_str = "\n".join(code_files)
        return code_str[:10000]  # limit length for safety

    except Exception as e:
        return f"Error fetching code: {e}"


# --------------------------
# Agent Functions
# --------------------------
def review_code(state: AgentState) -> AgentState:
    print("Agent 1: Reviewing repository code...")
    code = fetch_code(state["code_url"])

    prompt = f"""
    You are a senior application security analyst.
    Analyze the following **Java code** for security vulnerabilities.

    Provide:
    - Vulnerable code snippet or line
    - Java/JVM/JRE-specific code problem
    - Type of vulnerability (e.g., SQL Injection, XSS, Insecure Deserialization)
    - Severity (High/Medium/Low)
    - Why it is a problem
    - How to fix it briefly

    Code to review:
    {code}
    """

    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"code_content": code, "security_issues": resp.content}


def generate_fixes(state: AgentState) -> AgentState:
    print("Agent 2: Generating fixes and secure recommendations...")
    prompt = f"""
    You are a security remediation expert.
    Based on these findings, provide specific fixes and best practices.

    Security Issues:
    {state['security_issues']}

    For each issue, provide:
    - Specific code changes needed
    - Example of secure implementation
    - Short best-practice explanation
    """

    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"suggestions": resp.content}


def create_final_report(state: AgentState) -> AgentState:
    print("Agent 3: Creating final security report...")
    prompt = f"""
    Prepare a professional **Security Assessment Report** for:
    Repository: {state['code_url']}

    Include:
    1. Executive Summary
    2. List of Identified Vulnerabilities
    3. Severity Classification
    4. Recommendations and Fix Summary
    5. Implementation Roadmap

    Vulnerabilities:
    {state['security_issues']}

    Recommendations:
    {state['suggestions']}
    """

    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"final_report": resp.content}


# --------------------------
# Build Graph
# --------------------------
workflow = StateGraph(AgentState)

workflow.add_node("review_code", review_code)
workflow.add_node("generate_fixes", generate_fixes)
workflow.add_node("create_final_report", create_final_report)

workflow.set_entry_point("review_code")
workflow.add_edge("review_code", "generate_fixes")
workflow.add_edge("generate_fixes", "create_final_report")
workflow.add_edge("create_final_report", END)

app = workflow.compile()


# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    initial = {
        "code_url": "https://github.com/vulnerable-apps/verademo",
        "code_content": "",
        "security_issues": "",
        "suggestions": "",
        "final_report": ""
    }

    result = app.invoke(initial)
    print("\n" + "=" * 70)
    print(result["final_report"])
    print("=" * 70)
