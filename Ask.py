from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load FAISS index
index = faiss.read_index("faiss_index")

# Load chunks
with open("chunks.txt", "r", encoding="utf-8") as f:
    chunks = f.read().split("---CHUNK---")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Ask question
question = input("Ask a question: ")

# Convert question to embedding
question_embedding = model.encode([question])

# Search FAISS
D, I = index.search(np.array(question_embedding), k=3)

print("\nMost Relevant Answer:\n")

for idx in I[0]:
    if idx < len(chunks):
        print(chunks[idx])
        print("-" * 50)
