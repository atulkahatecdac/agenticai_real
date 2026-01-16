# pip install tavily-python langchain-tavily
# Also add TAVILY_API_KEY to .env file

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain.messages import SystemMessage, HumanMessage

from dotenv import load_dotenv

load_dotenv(override=True)

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.8
)

tool = TavilySearch(max_results=5, topic="general")

my_agent = create_agent(
    model=model,
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": "You are an AI assistant tasked with finding information about user's question and providing a concise response.",
            },
        ]
    )
)

response = my_agent.invoke(
    {"messages": 
        [HumanMessage
         ("Which is the most popular programming language now?")],
        "tools": [tool]}
)

ai_message = response['messages'][1].content  # AIMessage is usually the second message
print(ai_message)
