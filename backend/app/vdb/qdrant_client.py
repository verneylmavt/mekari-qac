"""Lazy, cache-only hybrid retrieval with bounded versioned evidence caches."""

import math
import re
import time
from collections import Counter, OrderedDict
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from threading import RLock

import numpy as np

from ..runtime import ServiceError
from .corpus import QUERY_INSTRUCTION, file_hash, load_corpus


class RetrievalUnavailable(RuntimeError):
    """A dependency or artifact is unavailable; expose only a safe message."""

    code = "retrieval_unavailable"


class DocumentRetriever:
    def __init__(self, settings, inference_gate=None):
        self.settings = settings
        self.inference_gate = inference_gate
        self._corpus = None
        self._client = None
        self._embedder = None
        self._reranker = None
        self._active_collection = None
        self._embedding_cache = OrderedDict()
        self._retrieval_cache = OrderedDict()
        self._lock = RLock()
        self._model_init_lock = RLock()
        self._lexical_index = None
        self._closed = False
        self._verified_snapshots = None
        self._request_deadline = ContextVar(f"retrieval_deadline_{id(self)}", default=None)

    def _deadline_middleware(self, request, call_next):
        # Qdrant's API timeout is server-side and integer rounded. Apply the exact
        # remaining budget to httpx as well, including metadata/alias requests.
        deadline = self._request_deadline.get()
        maximum = getattr(self.settings, "external_timeout_seconds", 30)
        self._check_deadline(deadline)
        timeout = deadline.timeout(maximum) if hasattr(deadline, "timeout") else maximum
        if deadline is not None and not hasattr(deadline, "timeout"):
            timeout = min(maximum, max(0.001, deadline - time.monotonic()))
        request.extensions["timeout"] = {
            key: timeout for key in ("connect", "read", "write", "pool")
        }
        result = call_next(request)
        self._check_deadline(deadline)
        return result

    def _check_deadline(self, deadline):
        if deadline is None:
            return
        if hasattr(deadline, "check"):
            deadline.check()
        elif time.monotonic() >= deadline:
            raise TimeoutError("Document retrieval deadline exceeded")

    def _load_corpus(self):
        with self._lock:
            if self._corpus is None:
                directory = Path(getattr(self.settings, "corpus_dir", "data/corpus"))
                self._corpus = load_corpus(directory, verify_sources=True)
        return self._corpus

    def _get_client(self):
        with self._lock:
            if self._client is None:
                from qdrant_client import QdrantClient

                self._client = QdrantClient(
                    url=self.settings.qdrant_url,
                    timeout=getattr(self.settings, "external_timeout_seconds", 30),
                    check_compatibility=False,
                )
                self._client.http.client.add_middleware(self._deadline_middleware)
        return self._client

    def _snapshot_paths(self):
        if self._verified_snapshots is not None:
            return self._verified_snapshots
        from huggingface_hub import snapshot_download

        profile = self._load_corpus().manifest["embedding_profile"]
        if (
            self.settings.embed_model_name != profile["model"]
            or self.settings.reranker_model_name != profile["reranker"]
        ):
            raise ValueError("Configured models do not match the corpus embedding profile")
        paths = []
        for model, revision, expected_hash in (
            (profile["model"], profile["model_revision"], profile["model_weights_sha256"]),
            (profile["reranker"], profile["reranker_revision"], profile["reranker_weights_sha256"]),
        ):
            path = Path(snapshot_download(model, revision=revision, local_files_only=True))
            required = ["config.json", "tokenizer_config.json", "tokenizer.json"]
            if model == profile["model"]:
                required += ["modules.json", "1_Pooling/config.json"]
            if not all((path / name).is_file() for name in required):
                raise ValueError("Required model snapshot is incomplete")
            weights = path / "model.safetensors"
            if not weights.is_file() or file_hash(weights) != expected_hash:
                raise ValueError("Cached model weights hash mismatch")
            paths.append(path)
        self._verified_snapshots = paths
        return paths

    def _validate_collection(self):
        corpus = self._load_corpus()
        client = self._get_client()
        aliases = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
        active = aliases.get(self.settings.qdrant_collection)
        expected = f"{self.settings.qdrant_collection}_{corpus.version}"
        if active != expected:
            raise ValueError("Active corpus alias version mismatch")
        info = client.get_collection(active)
        if (
            info.config.params.vectors.size != 768
            or info.config.params.vectors.distance != "Cosine"
        ):
            raise ValueError("Active corpus vector configuration mismatch")
        if client.count(active, exact=True).count != len(corpus.chunks):
            raise ValueError("Active corpus point count mismatch")
        # A versioned collection is validated once per lifetime; the name is immutable.
        if self._active_collection != active:
            for offset in range(0, len(corpus.chunks), 128):
                expected_chunks = corpus.chunks[offset : offset + 128]
                points = client.retrieve(
                    active,
                    ids=[c["chunk_id"] for c in expected_chunks],
                    with_payload=True,
                    with_vectors=True,
                )
                by_id = {str(p.id): p for p in points}
                if any(
                    c["chunk_id"] not in by_id
                    or by_id[c["chunk_id"]].payload != c
                    or not np.allclose(
                        by_id[c["chunk_id"]].vector, corpus.vectors[c["vector_row"]], atol=1e-6
                    )
                    for c in expected_chunks
                ):
                    raise ValueError("Active corpus payload/profile mismatch")
            with self._lock:
                self._active_collection = active
                self._retrieval_cache.clear()
        return active

    def ready(self):
        flags = {"qdrant_ok": False, "models_cached": False, "corpus_ok": False}
        try:
            if self._closed:
                return flags
            self._load_corpus()
            flags["corpus_ok"] = True
        except Exception:
            return flags
        try:
            self._snapshot_paths()
            flags["models_cached"] = True
        except Exception:
            pass
        try:
            self._validate_collection()
            flags["qdrant_ok"] = True
        except Exception:
            pass
        return flags

    def _ensure_models(self):
        with self._model_init_lock:
            if self._embedder is not None and self._reranker is not None:
                return
            import torch

            torch.set_num_threads(4)
            from sentence_transformers import CrossEncoder, SentenceTransformer

            embedding_path, reranking_path = self._snapshot_paths()
            if self._embedder is None:
                self._embedder = SentenceTransformer(
                    str(embedding_path), device="cpu", local_files_only=True
                )
            if self._reranker is None:
                self._reranker = CrossEncoder(
                    str(reranking_path),
                    device="cpu",
                    max_length=512,
                    local_files_only=True,
                    trust_remote_code=False,
                )

    def _infer(self, operation, deadline):
        self._check_deadline(deadline)
        result = (
            self.inference_gate.run(operation, deadline=deadline)
            if self.inference_gate
            else operation()
        )
        self._check_deadline(deadline)
        return result

    def _cached(self, cache, key, producer):
        with self._lock:
            if key in cache:
                cache.move_to_end(key)
                return deepcopy(cache[key])
        result = producer()
        with self._lock:
            cache[key] = deepcopy(result)
            cache.move_to_end(key)
            while len(cache) > getattr(self.settings, "retrieval_cache_size", 128):
                cache.popitem(last=False)
        return result

    def _embed_query(self, question, deadline):
        profile = self._load_corpus().manifest["profile_sha256"]

        def generate():
            def infer():
                self._ensure_models()
                return self._embedder.encode(
                    [QUERY_INSTRUCTION + question],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )[0]

            return self._infer(infer, deadline)

        return self._cached(
            self._embedding_cache,
            (self._corpus.version, self._active_collection, profile, question),
            generate,
        )

    def _dense_search(self, vector, document_id):
        from qdrant_client import models

        condition = (
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            )
            if document_id
            else None
        )
        deadline = self._request_deadline.get()
        maximum = getattr(self.settings, "external_timeout_seconds", 30)
        timeout = deadline.timeout(maximum) if hasattr(deadline, "timeout") else maximum
        response = self._get_client().query_points(
            self._active_collection,
            query=vector.tolist(),
            query_filter=condition,
            limit=getattr(self.settings, "retrieval_candidates", 30),
            with_payload=True,
            timeout=math.ceil(timeout),
        )
        return [p.payload for p in response.points]

    @staticmethod
    def _terms(text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def _lexical_search(self, question, document_id):
        corpus = self._load_corpus()
        if self._lexical_index is None:
            documents = [Counter(self._terms(c["text"])) for c in corpus.chunks]
            frequencies = Counter(term for document in documents for term in document)
            average = sum(sum(d.values()) for d in documents) / len(documents)
            self._lexical_index = (documents, frequencies, average)
        documents, frequencies, average = self._lexical_index
        query = set(self._terms(question))
        scores = []
        total = len(documents)
        for chunk, terms in zip(corpus.chunks, documents):
            if document_id and chunk["document_id"] != document_id:
                continue
            length = sum(terms.values())
            score = 0.0
            for term in query:
                frequency = terms[term]
                if frequency:
                    idf = math.log(
                        1 + (total - frequencies[term] + 0.5) / (frequencies[term] + 0.5)
                    )
                    score += (
                        idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / average))
                    )
            if score > 0:
                scores.append((score, chunk))
        scores.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
        return [c for _, c in scores[: getattr(self.settings, "retrieval_candidates", 30)]]

    def _rerank(self, question, chunks, deadline):
        def infer():
            self._ensure_models()
            # Reserve room for every full passage under the 512-token pair limit.
            tokenizer = getattr(self._reranker, "tokenizer", None)
            query = question
            if tokenizer is not None:
                query = tokenizer.decode(
                    tokenizer.encode(question, add_special_tokens=False)[:64],
                    skip_special_tokens=True,
                )
            return self._reranker.predict(
                [(query, c["text"]) for c in chunks], batch_size=16, show_progress_bar=False
            )

        scores = self._infer(infer, deadline)
        return sorted(
            zip(chunks, [float(s) for s in scores]),
            key=lambda pair: (-pair[1], pair[0]["chunk_id"]),
        )

    def retrieve(self, question, document_id=None, deadline=None):
        token = self._request_deadline.set(deadline)
        try:
            return self._retrieve(question, document_id, deadline)
        finally:
            self._request_deadline.reset(token)

    def _retrieve(self, question, document_id=None, deadline=None):
        if document_id not in (None, "bhatla", "eba"):
            raise ValueError("Unknown document filter")
        if not question or not question.strip():
            return []
        self._check_deadline(deadline)
        if self._closed:
            raise RetrievalUnavailable("Document retrieval is closed")
        try:
            self._load_corpus()
            if self._active_collection is None or self._client is not None:
                self._validate_collection()
            key = (
                self._corpus.version,
                self._active_collection,
                document_id,
                question,
                getattr(self.settings, "retrieval_final_chunks", 8),
                getattr(self.settings, "document_context_tokens", 6000),
            )

            def generate():
                vector = self._embed_query(question, deadline)
                candidates = self._dense_search(vector, document_id) + self._lexical_search(
                    question, document_id
                )
                unique = {}
                canonical = {c["chunk_id"]: c for c in self._corpus.chunks}
                for c in candidates:
                    known = canonical.get(c.get("chunk_id"))
                    if known is not None and (
                        document_id is None or known["document_id"] == document_id
                    ):
                        unique[known["chunk_id"]] = known
                if not unique:
                    return []
                ranked = self._rerank(question, list(unique.values()), deadline)
                output, tokens = [], 0
                for chunk, score in ranked:
                    if len(output) >= getattr(self.settings, "retrieval_final_chunks", 8):
                        break
                    if tokens + chunk["token_count"] > getattr(
                        self.settings, "document_context_tokens", 6000
                    ):
                        continue
                    output.append(
                        {
                            "citation_id": f"{chunk['document_id']}-p{chunk['page']}-{chunk['chunk_id'][:8]}",
                            "payload": deepcopy(chunk),
                            "rerank_score": score,
                        }
                    )
                    tokens += chunk["token_count"]
                return output

            result = self._cached(self._retrieval_cache, key, generate)
            self._check_deadline(deadline)
            return result
        except (TimeoutError, RetrievalUnavailable, ServiceError):
            raise
        except Exception as exc:
            self._check_deadline(deadline)
            raise RetrievalUnavailable("Document evidence is temporarily unavailable") from exc

    def close(self):
        with self._lock:
            if self._client is not None:
                self._client.close()
            self._client = self._embedder = self._reranker = None
            self._corpus = self._lexical_index = self._verified_snapshots = None
            self._embedding_cache.clear()
            self._retrieval_cache.clear()
            self._closed = True


def retrieve_relevant_chunks(query, top_k=8, use_reranker=True):
    """Compatibility helper; application resources should own DocumentRetriever instead."""
    from ..config import get_settings

    retriever = DocumentRetriever(get_settings())
    try:
        return retriever.retrieve(query)[:top_k]
    finally:
        retriever.close()
