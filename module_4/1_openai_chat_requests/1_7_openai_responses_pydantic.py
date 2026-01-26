# Structured output 

from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(override=True)

client = OpenAI()

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

response = client.responses.parse(
    model="gpt-4o-mini",
    input=[
        {   "role": "system", 
            "content": "Extract the event information."
        },
        {
            "role": "user",
            "content": "Alice and Bob are going to a picnic on Sunday.",
        },
    ],
    text_format=CalendarEvent, # Model must return the output as a CalendarEvent object
)

event = response.output_parsed # This is the CalendarEvent object, not a string, not a JSON

# Print structured result
print("Parsed Event (Pydantic object):", event)
print("Event name:", event.name)
print("Event date:", event.date)
print("Participants:", ", ".join(event.participants))

