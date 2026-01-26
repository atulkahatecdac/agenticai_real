# Analyze an image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI()

file = client.files.create(
    # file=open("C:\\code\\agenticai\\1_openai_chat_requests\\animals.pdf", "rb"),
    file=open("C:\\code\\agenticai\\1_openai_chat_requests\\cheque.pdf", "rb"),
    purpose="user_data"
)

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "file_id": file.id,
                },
                {
                    "type": "input_text",
                    # "text": "Which animals are these?",
                    "text": "Extract and list the various parts of the cheque."
                },
            ]
        }
    ]
)

print(response.output_text)