# backend/app/rag/qdrant_client.py

from typing import List, Dict, Any, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http.models import QueryResponse
from qdrant_client.http import models as qmodels

from ..config import get_settings

settings = get_settings()

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load embedding model (BGE)
_embed_model = SentenceTransformer(settings.embed_model_name, device=device)

# Load reranker model (BGE reranker)
_reranker = CrossEncoder(
    settings.reranker_model_name,
    device=device,
    max_length=512,
    trust_remote_code=True,
)

# Qdrant client
_qdrant_client = QdrantClient(url=settings.qdrant_url)
_collection_name = settings.qdrant_collection


def embed_texts(texts: List[str], batch_size: int = 32, is_query: bool = False) -> np.ndarray:
    if is_query:
        prefixed = [f"query: {t}" for t in texts]
    else:
        prefixed = [f"passage: {t}" for t in texts]

    embeddings = _embed_model.encode(
        prefixed,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query], batch_size=1, is_query=True)[0]


def search_qdrant(
    query: str,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    query_vector = embed_query(query)

    response: QueryResponse = _qdrant_client.query_points(
        collection_name=_collection_name,
        query=query_vector.tolist(),
        limit=top_k,
        with_payload=True,
    )

    output: List[Dict[str, Any]] = []
    for p in response.points:
        output.append(
            {
                "id": p.id,
                "score": p.score,
                "payload": p.payload,
            }
        )
    return output


def rerank_with_bge_reranker(
    query: str,
    retrieved_results: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not retrieved_results:
        return []

    texts = [r["payload"]["text"] for r in retrieved_results]
    pairs = [(query, t) for t in texts]

    scores = _reranker.predict(pairs)

    for r, s in zip(retrieved_results, scores):
        r["rerank_score"] = float(s)

    reranked = sorted(retrieved_results, key=lambda x: x["rerank_score"], reverse=True)

    if top_k is not None:
        reranked = reranked[:top_k]

    return reranked


def retrieve_relevant_chunks(
    query: str,
    top_k: int = 5,
    use_reranker: bool = True,
) -> List[Dict[str, Any]]:
    dense_results = search_qdrant(query, top_k=top_k)
    if not use_reranker:
        return dense_results
    return rerank_with_bge_reranker(query, dense_results, top_k=top_k)