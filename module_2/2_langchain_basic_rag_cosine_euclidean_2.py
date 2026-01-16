from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -----------------------------
# 1. Sentences (Loan example)
# -----------------------------
sentences = [
    # Same meaning, different length
    "Apply for loan",
    "I would like to apply for a personal loan as soon as possible",

    # Related but slightly different intent
    "Check loan application status",

    # Different domain
    "Reset my email password",
    "Wireless mouse stopped working suddenly"
]

# -----------------------------
# 2. Load Hugging Face model
# -----------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Each sentence → 384-dim vector
embeddings = model.encode(sentences)

print(f"Embedding shape: {embeddings.shape}")

# -----------------------------
# 3. Cosine Similarity
# -----------------------------
cos_sim = cosine_similarity(embeddings)

print("\n=== COSINE SIMILARITY ===\n")
for i in range(len(sentences)):
    for j in range(len(sentences)):
        print(f"S{i} vs S{j} = {cos_sim[i][j]:.3f}")
    print()

print("Sentence Index Reference:")
for i, s in enumerate(sentences):
    print(f"S{i}: {s}")

# -----------------------------
# 4. Euclidean Distance
# -----------------------------
euc_dist = euclidean_distances(embeddings)

print("\n=== EUCLIDEAN DISTANCE ===\n")
for i in range(len(sentences)):
    for j in range(len(sentences)):
        print(f"S{i} vs S{j} = {euc_dist[i][j]:.3f}")
    print()

# -----------------------------
# 5. Reduce to 2D (visualization only)
# -----------------------------
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(embeddings)

# -----------------------------
# 6. Plot
# -----------------------------
plt.figure()
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])

for i, sentence in enumerate(sentences):
    plt.text(
        vectors_2d[i, 0] + 0.01,
        vectors_2d[i, 1] + 0.01,
        f"S{i}"
    )

plt.title(
    "Sentence Embeddings (PCA Projection)\n"
    "Cosine → angle (meaning), Euclidean → distance (length-sensitive)"
)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
