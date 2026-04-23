"""
Run this ONCE before deploying to populate Pinecone with medical documents.
Usage: python ingest.py
Place your .txt files inside the docs/ folder.
"""

import os
import hashlib
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

DOCS_FOLDER = "docs"
CHUNK_SIZE = 200       # characters per chunk
OVERLAP = 40           # overlap between chunks
INDEX_NAME = "medical-rag"


def generate_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start = end - OVERLAP
    return [c for c in chunks if len(c) > 20]  # drop tiny trailing chunks


def ingest():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)

    if not os.path.exists(DOCS_FOLDER):
        print(f"Docs folder '{DOCS_FOLDER}' not found. Create it and add .txt files.")
        return

    for filename in os.listdir(DOCS_FOLDER):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(DOCS_FOLDER, filename)
        print(f"\nProcessing: {filename}")

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # Delete old chunks for this file
        try:
            index.delete(filter={"source": filename})
            print(f"  Deleted old vectors for {filename}")
        except Exception:
            print(f"  No existing data for {filename} (first run)")

        chunks = chunk_text(text)
        print(f"  Created {len(chunks)} chunks")

        embeddings = model.encode(chunks, show_progress_bar=True)

        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id": generate_id(chunk),
                "values": embedding.tolist(),
                "metadata": {
                    "text": chunk,
                    "source": filename,
                }
            })

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            index.upsert(vectors=vectors[i:i + batch_size])

        print(f"  Ingested {len(vectors)} vectors for {filename}")

    print("\nIngestion complete!")


if __name__ == "__main__":
    ingest()
