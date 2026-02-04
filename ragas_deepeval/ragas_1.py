# pip install datasets ragas langchain openai

# We need to convert our data into a Dataset object - this is the format that RAGAS expects
from datasets import Dataset
from ragas import evaluate # Main RAGAS evaluation function (Uses LLM internally)
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from dotenv import load_dotenv
load_dotenv(override=True)

data = {
    "question": [ # User input
        "Who invented the telephone?"
    ],
    "answer": [ # Mock LLM output
        "Alexander Graham Bell invented the telephone in 1876."
    ],
    "contexts": [[ # Documents retrieved by retriever (Vector DB/Knowledgebase)
        "Alexander Graham Bell is credited with inventing and patenting the telephone in 1876."
    ]],
    "reference": [ # Ground truth - Gold standard
        "Alexander Graham Bell invented the telephone."
    ]
}

dataset = Dataset.from_dict(data) # Convert to Dataset

# RAGAS reads each row: Q, A, C, R (See above)
# Each metric runs independently
# Result object will contain faithfulness, answer_relevancy, context_precision
# Example: faithfulness: 0.8, answer_relevancy: 0.9, context_precision: 0.7
# Meaning: 80% of the time, the model was correct, 90% of the time, the answer was relevant, 70% of the time, the context was relevant
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)

print(result)
