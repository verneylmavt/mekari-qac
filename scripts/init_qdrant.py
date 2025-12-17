# scripts/init_qdrant_bhatla.py

import os
import time
import subprocess
import json
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

def main():
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION", "bhatla_credit_fraud")

    base_dir = os.path.join("data", "Understanding Credit Card Frauds")
    chunks_path = os.path.join(base_dir, "Bhatla_chunks_ex.json")
    embeddings_path = os.path.join(base_dir, "Bhatla_embeddings_ex.npy")

    if not os.path.exists(chunks_path):
        raise FileNotFoundError(chunks_path)
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(embeddings_path)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunk_records = json.load(f)

    embeddings = np.load(embeddings_path)
    if len(chunk_records) != embeddings.shape[0]:
        raise ValueError("Number of chunks and embeddings do not match.")

    vector_dim = embeddings.shape[1]
    
    if subprocess.run(["docker", "start", "qdrant"]).returncode != 0:
        subprocess.run([
            "docker", "run", "-d",
            "--name", "qdrant",
            "-p", "6333:6333",
            # "-v", f"{os.getcwd()}\\qdrant:/qdrant/storage",
            "qdrant/qdrant:latest"
        ])

    client = QdrantClient(url=qdrant_url)
    print(f"Connected to Qdrant at {qdrant_url}")

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"Deleted old collection '{collection_name}'")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=vector_dim,
            distance=qmodels.Distance.COSINE,
        ),
    )
    print(f"Created collection '{collection_name}' with dim={vector_dim}")

    time.sleep(15)
    
    batch_size = 128
    total = len(chunk_records)
    uploaded = 0

    for i in range(0, total, batch_size):
        batch_records = chunk_records[i: i + batch_size]
        batch_vectors = embeddings[i: i + batch_size]

        points = []
        for rec, vec in zip(batch_records, batch_vectors):
            payload = {
                "chunk_id": rec["chunk_id"],
                "section": rec.get("section"),
                "subsection": rec.get("subsection"),
                "block_index": rec.get("block_index"),
                "chunk_index": rec.get("chunk_index"),
                "text": rec["text"],
            }
            points.append(
                qmodels.PointStruct(
                    id=rec["chunk_id"],
                    vector=vec.tolist(),
                    payload=payload,
                )
            )

        client.upsert(collection_name=collection_name, points=points)
        uploaded += len(batch_records)
        print(f"Uploaded {uploaded}/{total} chunks")

    print("Qdrant initialisation completed.")

if __name__ == "__main__":
    main()