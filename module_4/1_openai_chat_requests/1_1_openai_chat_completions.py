# pip install openai python-dotenv
# pip install -r requirements.txt

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

openai = OpenAI()

# First a very basic question
messages_created = [
    {
        "role": "system",
        "content": "You are a person who understands AI very well."
    },
    {
        "role": "user",
        "content": "Explain Generative AI for Dummies in 5 bullet points."
    },
    #{
    #    "role": "assistant",
    #    "content": "The factorial of 7 is 5040."
    #},
    #{
    #    "role": "user",
    #    "content": "Now explain how factorial works in simple terms."
    #}
]

response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages_created    
)

# print(response)
print(response.choices[0].message.content)
'''
# Now let us ask a tougher question
question = "Please propose a hard, challenging question to assess someone's IQ. Respond only with the question."
messages = [{"role": "user", "content": question}]

response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

# Save the question returned by the LLM into a variable called question
question = response.choices[0].message.content

print(f"OpenAI Question: {question}")  

# now form a new message list
messages = [{
             "role": "user", 
			 "content": question
		   }]

response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

answer = response.choices[0].message.content
print(f"GPT answer: {answer} ")

'''