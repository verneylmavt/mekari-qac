"""Validated, immutable corpus artifacts shared by setup and runtime."""

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NAMESPACE = uuid.UUID("a70a6677-b0c9-5b19-8d15-b6c6fd804292")
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_chunk_id(chunk):
    identity = {
        k: chunk[k]
        for k in ("document_id", "source_sha256", "page", "section", "chunk_index", "text")
    }
    return str(uuid.uuid5(NAMESPACE, canonical_hash(identity)))


@dataclass(frozen=True)
class Corpus:
    manifest: dict
    chunks: list
    vectors: np.ndarray

    @property
    def version(self):
        return self.manifest["artifact_version"]


def load_corpus(directory, *, verify_sources=False, source_root=None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported corpus manifest schema")
    profile = manifest["embedding_profile"]
    if profile["passage_instruction"] != "" or profile["query_instruction"] != QUERY_INSTRUCTION:
        raise ValueError("Legacy or incompatible embedding profile")
    if (
        profile["model"] != "BAAI/bge-base-en-v1.5"
        or profile["dimension"] != 768
        or not profile["normalized"]
    ):
        raise ValueError("Incompatible embedding model profile")
    if manifest["profile_sha256"] != canonical_hash(profile):
        raise ValueError("Embedding profile hash mismatch")
    for name in ("chunks.json", "vectors.npy"):
        if file_hash(directory / name) != manifest["artifacts"][name]["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {name}; download Git LFS objects if needed")
    chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(directory / "vectors.npy", allow_pickle=False)
    if vectors.dtype != np.float32 or vectors.shape != (len(chunks), profile["dimension"]):
        raise ValueError("Vector row count, dimension or dtype mismatch")
    if len(chunks) != manifest["chunk_count"] or not chunks:
        raise ValueError("Chunk count mismatch")
    if not np.isfinite(vectors).all() or not np.allclose(
        np.linalg.norm(vectors, axis=1), 1, atol=1e-4
    ):
        raise ValueError("Vectors must be finite and normalized")
    ids = [c["chunk_id"] for c in chunks]
    if len(set(ids)) != len(ids) or canonical_hash(ids) != manifest["point_order_sha256"]:
        raise ValueError("Point row ordering hash mismatch or duplicate IDs")
    for row, chunk in enumerate(chunks):
        if chunk["chunk_id"] != stable_chunk_id(chunk) or chunk["vector_row"] != row:
            raise ValueError("Stable ID or vector row mismatch")
        source = manifest["sources"][chunk["document_id"]]
        if chunk["source_sha256"] != source["sha256"] or not 1 <= chunk["page"] <= source["pages"]:
            raise ValueError("Invalid source provenance")
        if (
            not chunk["text"].strip()
            or not 0 < chunk["token_count"] <= manifest["chunking"]["max_tokens"]
        ):
            raise ValueError("Empty or oversized chunk")
        if (
            chunk["profile_sha256"] != manifest["profile_sha256"]
            or chunk["artifact_version"] != manifest["artifact_version"]
        ):
            raise ValueError("Chunk artifact version mismatch")
    if manifest["vectors_sha256"] != manifest["artifacts"]["vectors.npy"]["sha256"]:
        raise ValueError("Vector artifact version hash mismatch")
    identity = {
        k: manifest[k]
        for k in ("profile_sha256", "point_order_sha256", "sources", "chunking", "vectors_sha256")
    }
    if manifest["artifact_version"] != canonical_hash(identity)[:16]:
        raise ValueError("Artifact version hash mismatch")
    if verify_sources:
        root = (
            Path(source_root).resolve()
            if source_root is not None
            else Path(__file__).resolve().parents[3]
        )
        for source in manifest["sources"].values():
            if file_hash(root / source["path"]) != source["sha256"]:
                raise ValueError("Source PDF hash mismatch")
    vectors.flags.writeable = False
    return Corpus(manifest, chunks, vectors)
