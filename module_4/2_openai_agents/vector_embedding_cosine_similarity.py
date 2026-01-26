from sentence_transformers import SentenceTransformer, util
import numpy as np

# Two sentences with similar meaning but different wording
sentence_1 = "A man is playing a game of football."
sentence_2 = "Someone is strumming a musical instrument."
# sentence_2 = "Soccer is a popular sport."

# Load the pretrained model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Encode both sentences into embeddings
embedding_1 = model.encode(sentence_1, convert_to_tensor=True)
embedding_2 = model.encode(sentence_2, convert_to_tensor=True)

# Calculate cosine similarity
cos_sim = util.pytorch_cos_sim(embedding_1, embedding_2)

# Print the similarity score
print(f"Cosine similarity between the sentences: {cos_sim.item():.4f}")
print(embedding_1)
print(embedding_1.shape)
