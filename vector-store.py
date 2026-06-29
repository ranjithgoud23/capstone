from sentence_transformers import SentenceTransformer
import faiss

# Read extracted text
with open("pdf_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split into chunks
chunk_size = 500
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

print(f"Total chunks: {len(chunks)}")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(chunks)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# Save index
faiss.write_index(index, "faiss_index")

# Save chunks
with open("chunks.txt", "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk.replace("\n", " ") + "\n---CHUNK---\n")

print("FAISS index created successfully")