from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend.app.vdb.corpus import load_corpus
from backend.app.vdb.qdrant_client import DocumentRetriever

ROOT = Path(__file__).resolve().parents[1]


class Embedder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.extend(texts)
        return np.ones((len(texts), 768), dtype=np.float32) / np.sqrt(768)


class Reranker:
    def predict(self, pairs, **kwargs):
        return [10 if "71%" in p[1] else 1 for p in pairs]


def retriever():
    settings = SimpleNamespace(
        corpus_dir=ROOT / "data/corpus",
        qdrant_collection="fraud_documents",
        retrieval_candidates=30,
        retrieval_final_chunks=8,
        document_context_tokens=6000,
        retrieval_cache_size=2,
        embed_model_name="BAAI/bge-base-en-v1.5",
        reranker_model_name="BAAI/bge-reranker-base",
    )
    r = DocumentRetriever(settings)
    r._corpus = load_corpus(settings.corpus_dir)
    r._embedder = Embedder()
    r._reranker = Reranker()
    r._active_collection = "test-version"
    r._dense_search = lambda vector, document_id: []
    return r


def test_lexical_candidates_document_filter_and_whole_chunk_caps():
    r = retriever()
    output = r.retrieve("cross border card fraud 71%", document_id="eba")
    assert output and all(c["payload"]["document_id"] == "eba" for c in output)
    assert len(output) <= 8
    assert len({c["payload"]["chunk_id"] for c in output}) == len(output)
    assert sum(c["payload"]["token_count"] for c in output) <= 6000
    assert all(
        c["payload"]["text"]
        == next(x["text"] for x in r._corpus.chunks if x["chunk_id"] == c["payload"]["chunk_id"])
        for c in output
    )
    assert r._embedder.calls[0].startswith(
        "Represent this sentence for searching relevant passages: "
    )


def test_retrieval_cache_is_filter_and_version_keyed_and_bounded():
    r = retriever()
    first = r.retrieve("fraud", "eba")
    first[0]["payload"]["text"] = "mutated"
    assert r.retrieve("fraud", "eba")[0]["payload"]["text"] != "mutated"
    r.retrieve("fraud", "bhatla")
    r._active_collection = "new-version"
    r.retrieve("fraud", "eba")
    assert len(r._retrieval_cache) <= 2
    assert len(r._embedder.calls) == 2


def test_context_budget_never_slices_a_chunk():
    r = retriever()
    r.settings.document_context_tokens = 1
    assert r.retrieve("credit card fraud", "eba") == []


def test_deadline_and_inference_saturation_propagate_safe_errors():
    import pytest

    from backend.app.runtime import Deadline, InferenceGate, ServiceError

    r = retriever()
    with pytest.raises(ServiceError) as expired:
        r.retrieve("fraud", deadline=Deadline(0))
    assert expired.value.status_code == 504
    gate = InferenceGate(1)
    r.inference_gate = gate
    gate._permits.acquire()
    try:
        with pytest.raises(ServiceError) as busy:
            r.retrieve("fraud")
        assert busy.value.code == "inference_busy"
    finally:
        gate._permits.release()


def test_dense_and_lexical_candidates_are_unioned_before_reranking():
    r = retriever()
    dense = next(
        c
        for c in r._corpus.chunks
        if c["document_id"] == "eba" and c["page"] == 6 and "71%" in c["text"]
    )
    r._dense_search = lambda vector, document_id: [dense, dense]
    output = r.retrieve("cross border 71%", "eba")
    assert dense["chunk_id"] in {c["payload"]["chunk_id"] for c in output}
    assert len({c["payload"]["chunk_id"] for c in output}) == len(output)


def test_http_deadline_limits_metadata_and_query_transport_timeouts():
    import httpx

    from backend.app.runtime import Deadline

    r = retriever()
    token = r._request_deadline.set(Deadline(3))
    try:
        request = httpx.Request("GET", "http://qdrant.test/aliases")
        result = r._deadline_middleware(request, lambda req: req.extensions["timeout"])
        assert all(0 < duration <= 3 for duration in result.values())
    finally:
        r._request_deadline.reset(token)


def test_readiness_flags_and_alias_changes_are_checked_before_cache_use():
    import pytest
    from qdrant_client import QdrantClient, models

    from backend.app.vdb.qdrant_client import RetrievalUnavailable
    from scripts.init_qdrant import initialize

    r = retriever()
    client = QdrantClient(":memory:")
    initialize(client, ROOT / "data/corpus")
    r._client = client
    r._active_collection = None

    def absent_models():
        raise ValueError("test has no cached models")

    r._snapshot_paths = absent_models
    flags = r.ready()
    assert flags == {"corpus_ok": True, "qdrant_ok": True, "models_cached": False}
    assert r.retrieve("cross border fraud", "eba")
    client.create_collection(
        "different", vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
    )
    client.update_collection_aliases(
        [
            models.DeleteAliasOperation(
                delete_alias=models.DeleteAlias(alias_name="fraud_documents")
            ),
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name="different", alias_name="fraud_documents"
                )
            ),
        ]
    )
    with pytest.raises(RetrievalUnavailable):
        r.retrieve("cross border fraud", "eba")
    r.close()


def test_runtime_rejects_tampered_normalized_vectors_before_first_retrieval():
    import pytest
    from qdrant_client import QdrantClient, models

    from backend.app.vdb.qdrant_client import RetrievalUnavailable
    from scripts.init_qdrant import initialize

    r = retriever()
    client = QdrantClient(":memory:")
    name = initialize(client, ROOT / "data/corpus")
    chunk = r._corpus.chunks[0]
    corrupt = [0.0] * 768
    corrupt[0] = 1.0
    client.upsert(
        name,
        points=[models.PointStruct(id=chunk["chunk_id"], vector=corrupt, payload=chunk)],
        wait=True,
    )
    r._client, r._active_collection = client, None
    with pytest.raises(RetrievalUnavailable):
        r.retrieve("fraud")
    r.close()


def test_concurrent_cold_model_initialization_allocates_each_model_once(monkeypatch):
    import sys
    import time
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Lock

    r = retriever()
    r._embedder = r._reranker = None
    r._snapshot_paths = lambda: (Path("embedding"), Path("reranking"))
    counts, counts_lock = Counter(), Lock()

    def construct(name, *args, **kwargs):
        with counts_lock:
            counts[name] += 1
        time.sleep(0.03)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=lambda *a, **kw: construct("embedding", *a, **kw),
            CrossEncoder=lambda *a, **kw: construct("reranking", *a, **kw),
        ),
    )
    barrier = Barrier(2)

    def initialize():
        barrier.wait()
        r._ensure_models()

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda _: initialize(), range(2)))
    assert counts == {"embedding": 1, "reranking": 1}
