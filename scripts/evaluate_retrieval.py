"""Offline retrieval benchmark using real cached models and local in-memory Qdrant.

This evaluates evidence retrieval, not live LLM answer accuracy. No provider calls.
"""

import argparse
import json
import sys
import time
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient  # noqa: E402

from backend.app.config import Settings  # noqa: E402
from backend.app.runtime import Deadline, InferenceGate  # noqa: E402
from backend.app.vdb.corpus import load_corpus  # noqa: E402
from backend.app.vdb.qdrant_client import DocumentRetriever  # noqa: E402
from scripts.init_qdrant import initialize  # noqa: E402


def evaluate(output):
    settings = Settings(_env_file=None)
    corpus = load_corpus(settings.corpus_dir, verify_sources=True)
    questions = json.loads((ROOT / "assets/evaluation/questions.json").read_text())
    results = []
    with closing(QdrantClient(":memory:")) as client:
        initialize(client, settings.corpus_dir, settings.qdrant_collection)
        retriever = DocumentRetriever(settings, InferenceGate(1))
        retriever._client = client
        for question in questions:
            if not question.get("pages"):
                continue
            start = time.monotonic()
            chunks = retriever.retrieve(
                question["question"], question["document_id"], Deadline(120)
            )
            ranks = [
                i + 1
                for i, chunk in enumerate(chunks)
                if chunk["payload"]["page"] in question["pages"]
            ]
            joined = "\n".join(c["payload"]["text"] for c in chunks).lower()
            terms = {term: term.lower() in joined for term in question["gold_terms"]}
            results.append(
                {
                    "id": question["id"],
                    "hit_at_8": bool(ranks),
                    "reciprocal_rank": 1 / min(ranks) if ranks else 0,
                    "gold_terms_present": terms,
                    "citations": [c["citation_id"] for c in chunks],
                    "seconds": round(time.monotonic() - start, 3),
                }
            )
        # The client is owned by this context, not the injected retriever.
        retriever._client = None
        retriever.close()
    report = {
        "mode": "offline_real_retrieval",
        "artifact_version": corpus.version,
        "scope": "Seven documentary probes. Does not measure planner, SQL generation or live answer accuracy.",
        "cases": len(results),
        "hit_at_8": sum(r["hit_at_8"] for r in results) / len(results),
        "mean_reciprocal_rank": sum(r["reciprocal_rank"] for r in results) / len(results),
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "assets/evaluation/retrieval_report.json"
    )
    evaluate(parser.parse_args().output)
