# pip install langchain-huggingface langchain-chroma sentence-transformers pypdf
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 1. Initialize Hugging Face Embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True} # Essential for clean cosine similarity
)

persist_path = r"c:/code/agenticai_realpage/module_2/local_chroma_db"

# 2. Initialize the Vector Store
vector_db = Chroma(
    persist_directory=persist_path,
    embedding_function=hf_embeddings
)

# 3. Populate if empty
if len(vector_db.get()['ids']) == 0:
    actual_sentences = [
        "The introduction provides an overview of the artificial intelligence landscape in 2024.",
        "Sustainable energy sources like solar and wind are crucial for reducing carbon emissions.",
        "Deep learning models require large amounts of data to achieve high accuracy in image recognition.",
        "The history of Rome spans over two thousand years, beginning as a small Italian village.",
        "Regular exercise and a balanced diet are key components of maintaining long-term physical health.",
        "Quantum computing uses qubits to perform complex calculations much faster than classical computers.",
        "Effective communication is the foundation of successful project management in corporate environments.",
        "The Great Barrier Reef is the world's largest coral reef system, located in Australia.",
        "Python is a versatile programming language widely used for data science and web development.",
        "Recent advancements in space exploration have made Mars colonization a topic of serious debate."
    ]
    dummy_docs = [Document(page_content=text) for text in actual_sentences]
    vector_db.add_documents(dummy_docs)

# 4. Perform Search with Relevance Scores
query = "What does the document say about the introduction?"
# This returns a list of tuples: (Document, Score)
results_with_scores = vector_db.similarity_search_with_relevance_scores(query, k=3)

print(f"Query: {query}\n")
print(f"{'Score (Cosine)':<15} | {'Content'}")
print("-" * 60)

for doc, score in results_with_scores:
    print(f"{score:<15.4f} | {doc.page_content}")