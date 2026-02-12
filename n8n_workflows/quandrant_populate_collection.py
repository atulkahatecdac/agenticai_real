# pip install qdrant-client openai

# pip install qdrant-client openai

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)


# Initialize clients
client = QdrantClient(host="localhost", port=6333)
openai_client = OpenAI()

# Q&A dataset
faq_data = [
    {
        "question": "What is the rate of interest?",
        "answer": "Currently, we offer a rate of interest of 3% PA on savings bank accounts."
    },
    {
        "question": "How can I open an account?",
        "answer": "You can open an account online using Aadhaar and PAN."
    },
    {
        "question": "What is the minimum balance?",
        "answer": "The minimum balance required is ₹10,000."
    }
]

points = []

for i, item in enumerate(faq_data):
    
    # Generate embedding from QUESTION
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=item["question"]
    )
    
    vector = response.data[0].embedding

    points.append(
        PointStruct(
            id=i,
            vector=vector,
            payload={
                "question": item["question"],
                "answer": item["answer"]
            }
        )
    )

# Insert into Qdrant
client.upsert(
    collection_name="customer_queries",
    points=points
)

print("Q&A data inserted!")
