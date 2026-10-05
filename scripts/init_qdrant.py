"""Validate artifacts, upload a new collection, then atomically switch an alias.

Never starts infrastructure or deletes an existing collection.
"""

import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.vdb.corpus import load_corpus  # noqa: E402 - support direct script execution


def initialize(client, corpus_dir, alias="fraud_documents"):
    corpus = load_corpus(corpus_dir, verify_sources=True)
    import numpy as np
    from qdrant_client import models

    name = f"{alias}_{corpus.version}"
    aliases = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
    if client.collection_exists(alias) and alias not in aliases:
        raise ValueError(
            "The alias name is occupied by a collection; choose a new alias explicitly"
        )
    if not client.collection_exists(name):
        client.create_collection(
            name, vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
        )
    if aliases.get(alias) != name:
        for offset in range(0, len(corpus.chunks), 128):
            points = [
                models.PointStruct(
                    id=c["chunk_id"], vector=corpus.vectors[c["vector_row"]].tolist(), payload=c
                )
                for c in corpus.chunks[offset : offset + 128]
            ]
            client.upsert(name, points=points, wait=True)
    info = client.get_collection(name)
    if (
        info.config.params.vectors.size != 768
        or info.config.params.vectors.distance != models.Distance.COSINE
    ):
        raise ValueError("Uploaded collection vector configuration mismatch")
    if client.count(name, exact=True).count != len(corpus.chunks):
        raise ValueError("Uploaded collection count mismatch; active alias was not changed")
    for offset in range(0, len(corpus.chunks), 128):
        expected = corpus.chunks[offset : offset + 128]
        stored = client.retrieve(
            name, ids=[c["chunk_id"] for c in expected], with_payload=True, with_vectors=True
        )
        by_id = {str(p.id): p for p in stored}
        for chunk in expected:
            point = by_id.get(chunk["chunk_id"])
            if (
                point is None
                or point.payload != chunk
                or not np.allclose(point.vector, corpus.vectors[chunk["vector_row"]], atol=1e-6)
            ):
                raise ValueError(
                    "Uploaded point provenance/vector mismatch; active alias was not changed"
                )
    if aliases.get(alias) != name:
        operations = []
        if alias in aliases:
            operations.append(
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=name, alias_name=alias)
            )
        )
        client.update_collection_aliases(operations)
    return name


def main():
    from qdrant_client import QdrantClient

    from backend.app.config import get_settings

    settings = get_settings()
    directory = Path(getattr(settings, "corpus_dir", ROOT / "data/corpus"))
    load_corpus(directory, verify_sources=True)
    with closing(QdrantClient(url=settings.qdrant_url, timeout=30)) as client:
        name = initialize(client, directory, settings.qdrant_collection)
    print(f"Validated and activated {name}; previous collections retained")


if __name__ == "__main__":
    main()
