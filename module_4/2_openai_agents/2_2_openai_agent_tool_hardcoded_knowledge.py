from openai import OpenAI
from dotenv import load_dotenv
import asyncio
from agents import Agent, Runner, function_tool

load_dotenv(override=True)

client = OpenAI()

knowledge_base = {
    "shipping_time": "Our standard shipping time is 3-5 business days.",
    "return_policy": "You can return any product within 30 days of delivery.",
    "warranty": "All products come with a one-year warranty covering manufacturing defects.",
    "payment_methods": "We accept credit cards, debit cards, and PayPal.",
    "customer_support": "You can reach our support team 24/7 via email or chat."
}

@function_tool
async def faq_invoker(topic: str) -> str:
    """Provides answers to frequently asked customer support questions."""
    user_query = topic.lower()
    for topic_key, answer in knowledge_base.items():
        if topic_key in user_query:
            return answer

    fallback = (
        "I'm sorry, but I couldn't find specific information about that topic. "
        "Please check the company's website or contact customer support directly."
    )
    return fallback

faq_agent = Agent(
    name="Customer Support Bot",
    instructions=(
        "You are a helpful customer support assistant. "
        "Answer questions using your FAQ tool when appropriate."
    ),
    tools=[faq_invoker]
)

async def chat_with_support(message):
    session = await Runner.run(faq_agent, message)
    return session.final_output

async def main():
    print("Customer Support Bot is running. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting.")
            break
        response = await chat_with_support(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    asyncio.run(main())
