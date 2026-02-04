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
    topic: str
    headlines: str
    summary: str
    sentiment: str
    final_report: str
    sentiment_label: str

# --------------------------
# Setup
# --------------------------
load_dotenv(override=True)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
news_key = os.getenv("NEWS_API_KEY")

# --------------------------
# Agent Functions
# --------------------------
def fetch_news(state: AgentState) -> AgentState:
    url = f"https://newsapi.org/v2/everything?q={state['topic']}&apiKey={news_key}&pageSize=5"
    try:
        r = requests.get(url)
        articles = [a["title"] for a in r.json().get("articles", [])]
        headlines = "\n".join(articles)
    except Exception as e:
        headlines = f"Error fetching news: {e}"
    return {"headlines": headlines}

def summarize_news(state: AgentState) -> AgentState:
    prompt = f"Summarize these headlines about {state['topic']}:\n{state['headlines']}"
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"summary": resp.content}

def analyze_sentiment(state: AgentState) -> AgentState:
    prompt = f"Is this summary overall positive or negative?\n\n{state['summary']}"
    resp = llm.invoke([HumanMessage(content=prompt)])
    sentiment = resp.content.strip().lower()
    label = "positive" if "positive" in sentiment else "negative"
    return {"sentiment": sentiment, "sentiment_label": label}

def investor_summary(state: AgentState) -> AgentState:
    prompt = f"Write an investor-focused insight based on:\n{state['summary']}"
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"final_report": resp.content}

def public_summary(state: AgentState) -> AgentState:
    prompt = f"Write a short public news digest based on:\n{state['summary']}"
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"final_report": resp.content}

# --------------------------
# Build Graph
# --------------------------
workflow = StateGraph(AgentState)
workflow.add_node("fetch_news", fetch_news)
workflow.add_node("summarize_news", summarize_news)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("investor_summary", investor_summary)
workflow.add_node("public_summary", public_summary)

workflow.set_entry_point("fetch_news")
workflow.add_edge("fetch_news", "summarize_news")
workflow.add_edge("summarize_news", "analyze_sentiment")
workflow.add_conditional_edges(
    "analyze_sentiment",
    lambda state: "investor_summary" if state["sentiment_label"] == "positive" else "public_summary"
)
workflow.add_edge("investor_summary", END)
workflow.add_edge("public_summary", END)

app = workflow.compile()

# --------------------------
# Streaming function
# --------------------------
def run_agent_for_topic(topic: str):
    initial = {
        "topic": topic,
        "headlines": "",
        "summary": "",
        "sentiment": "",
        "final_report": "",
        "sentiment_label": ""
    }

    # 1. Fetch news
    result = app.invoke({**initial})
    headlines = result["headlines"]
    yield f"Headlines:\n{headlines}\n\n"

    # 2. Summarize
    summary_result = summarize_news({**initial, "headlines": headlines, "topic": topic})
    summary = summary_result["summary"]
    yield f"Summary:\n{summary}\n\n"

    # 3. Sentiment
    sentiment_result = analyze_sentiment({**initial, "summary": summary})
    sentiment = sentiment_result["sentiment"]
    label = sentiment_result["sentiment_label"]
    yield f"Sentiment: {sentiment} ({label})\n\n"

    # 4. Final report
    if label == "positive":
        final = investor_summary({**initial, "summary": summary})["final_report"]
        yield f"Investor Summary:\n{final}\n"
    else:
        final = public_summary({**initial, "summary": summary})["final_report"]
        yield f"Public Summary:\n{final}\n"
