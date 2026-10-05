import json
from pathlib import Path

import pytest

from backend.app.vdb.corpus import load_corpus, stable_chunk_id
from scripts.init_qdrant import initialize

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_corpus_integrity_and_gold_facts():
    corpus = load_corpus(ROOT / "data/corpus")
    assert {c["document_id"] for c in corpus.chunks} == {"bhatla", "eba"}
    assert corpus.manifest["sources"]["bhatla"]["pages"] == 17
    assert corpus.manifest["sources"]["eba"]["pages"] == 35
    assert corpus.vectors.shape == (len(corpus.chunks), 768)
    for chunk in corpus.chunks:
        assert chunk["chunk_id"] == stable_chunk_id(chunk)
        assert chunk["token_count"] <= 420
        assert chunk["source_sha256"] == corpus.manifest["sources"][chunk["document_id"]]["sha256"]

    def page(doc, number):
        return "\n".join(
            c["text"] for c in corpus.chunks if c["document_id"] == doc and c["page"] == number
        )

    bhatla = page("bhatla", 4)
    import re

    for mechanism, percentage in (
        ("Lost or stolen card", 48),
        ("Identity theft", 15),
        ("Skimming (or cloning)", 14),
        ("Counterfeit card", 12),
        ("Mail intercept fraud", 6),
        ("Other", 5),
    ):
        assert re.search(re.escape(mechanism) + r"\s+" + str(percentage) + "%", bhatla)
    assert "Celent" in bhatla
    assert "0.08%" in page("bhatla", 15) and "0.06%" in page("bhatla", 15)
    assert "ten times" in page("eba", 6)
    assert "71% in value terms in H1 2023" in page("eba", 6)
    assert "71% of the total value" in page("eba", 27)
    assert "68% of the total volume" in page("eba", 27)


def test_row_reordering_rejected_before_remote_mutation(tmp_path):
    source = ROOT / "data/corpus"
    for name in ("manifest.json", "chunks.json", "vectors.npy"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    chunks = json.loads((tmp_path / "chunks.json").read_text())
    chunks[0], chunks[1] = chunks[1], chunks[0]
    (tmp_path / "chunks.json").write_text(json.dumps(chunks))

    class NoRemote:
        def __getattr__(self, name):
            pytest.fail(f"Remote operation before artifact validation: {name}")

    with pytest.raises(ValueError, match="hash"):
        initialize(NoRemote(), tmp_path)


def test_initializer_atomically_replaces_alias_and_keeps_previous_collection():
    from qdrant_client import QdrantClient, models

    client = QdrantClient(":memory:")
    client.create_collection(
        "previous", vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
    )
    client.update_collection_aliases(
        [
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name="previous", alias_name="fraud_documents"
                )
            )
        ]
    )
    name = initialize(client, ROOT / "data/corpus")
    assert client.collection_exists("previous")
    assert client.count("fraud_documents", exact=True).count == len(
        load_corpus(ROOT / "data/corpus").chunks
    )
    assert client.get_aliases().aliases[0].collection_name == name
    assert initialize(client, ROOT / "data/corpus") == name
    client.close()


def test_stable_ids_change_with_page_text_or_source_but_not_vector_row():
    corpus = load_corpus(ROOT / "data/corpus")
    chunk = corpus.chunks[0]
    assert stable_chunk_id({**chunk, "vector_row": 999}) == chunk["chunk_id"]
    for key, changed in (
        ("text", chunk["text"] + " changed"),
        ("page", chunk["page"] + 1),
        ("source_sha256", "a" * 64),
    ):
        assert stable_chunk_id({**chunk, key: changed}) != chunk["chunk_id"]


def test_failed_upload_retains_alias_and_can_resume_without_deleting_collections():
    from qdrant_client import QdrantClient, models

    client = QdrantClient(":memory:")
    client.create_collection(
        "previous", vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
    )
    client.update_collection_aliases(
        [
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name="previous", alias_name="fraud_documents"
                )
            )
        ]
    )
    original = client.upsert

    def interrupted(name, *, points, wait):
        original(name, points=points[:1], wait=wait)
        raise RuntimeError("simulated interrupted upload")

    client.upsert = interrupted
    with pytest.raises(RuntimeError, match="interrupted"):
        initialize(client, ROOT / "data/corpus")
    assert client.get_aliases().aliases[0].collection_name == "previous"
    client.upsert = original
    assert initialize(client, ROOT / "data/corpus").startswith("fraud_documents_")
    assert client.collection_exists("previous")
    client.close()


def test_corpus_can_be_validated_outside_repository_data_directory(tmp_path):
    for name in ("manifest.json", "chunks.json", "vectors.npy"):
        (tmp_path / name).write_bytes((ROOT / "data/corpus" / name).read_bytes())
    corpus = load_corpus(tmp_path, verify_sources=True)
    assert corpus.version == load_corpus(ROOT / "data/corpus").version
