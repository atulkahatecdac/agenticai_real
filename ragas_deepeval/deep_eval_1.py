# pip install deepeval
import os
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric

from dotenv import load_dotenv
load_dotenv(override=True)

# 1. Setup the metrics (equivalent to RAGAS metrics)
# You can set thresholds: if the score is below this, the "test" fails.
faithfulness = FaithfulnessMetric(threshold=0.7)
relevancy = AnswerRelevancyMetric(threshold=0.7)
precision = ContextualPrecisionMetric(threshold=0.7)

# 2. Define the Test Case 
# DeepEval uses a specific object structure for each evaluation point
test_case = LLMTestCase(
    input="Who invented the telephone?",
    actual_output="Alexander Graham Bell invented the telephone in 1876.",
    retrieval_context=["Alexander Graham Bell is credited with inventing and patenting the telephone in 1876."],
    expected_output="Alexander Graham Bell invented the telephone."
)

# 3. Execute the evaluation
# Option A: Measure metrics individually
faithfulness.measure(test_case)
relevancy.measure(test_case)
precision.measure(test_case)

print(f"Faithfulness Score: {faithfulness.score}")
print(f"Relevancy Score: {relevancy.score}")
print(f"Precision Score: {precision.score}")
print(f"Reasoning: {faithfulness.reason}")