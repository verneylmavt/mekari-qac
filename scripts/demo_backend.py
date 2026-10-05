"""Clearly labelled offline UI fixture. No real warehouse, inference or API calls.

Run: python -m uvicorn scripts.demo_backend:app --port 8001
"""

from types import SimpleNamespace

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.vdb.corpus import load_corpus


class FixtureLLM:
    def complete(self, task, schema, system, user, deadline):
        import json

        content = json.loads(user)
        if task == "planner":
            question = content["question"]
            document = any(
                word in question.lower() for word in ("eba", "bhatla", "mechanism", "cross-border")
            )
            data = any(
                word in question.lower()
                for word in ("monthly", "dataset", "transaction", "categories")
            )
            route = "mixed" if document and data else "document" if document else "data"
            return schema.model_validate(
                {
                    "route": route,
                    "standalone_question": question,
                    "document_id": content["document_id"]
                    or ("bhatla" if "bhatla" in question.lower() else "eba"),
                    "data_question": question if data or not document else "",
                    "document_question": question if document else "",
                    "clarification": "",
                }
            )
        if task == "sql":
            return schema.model_validate(
                {
                    "sql": "SELECT year_month, total_tx, fraud_tx, fraud_rate "
                    "FROM agg_monthly_fraud ORDER BY year_month"
                }
            )
        if task == "answer":
            evidence = json.loads(content["evidence"])
            answers, citations = [], []
            if "warehouse" in evidence:
                answers.append(
                    "Offline demo fixture: the synthetic monthly fraud rate falls from "
                    "2% to 1%, then rises to 1.5%. These rows are illustrative fixture data. [SQL]"
                )
                citations.append("SQL")
            if "documents" in evidence:
                doc = evidence["documents"][0]
                reference = doc["citation_id"]
                if doc["document_id"] == "eba":
                    answers.append(
                        "In H1 2023, cross-border transactions accounted for 71% of "
                        "card fraud value and 68% of card fraud volume in the EBA report's population. "
                        f"This offline fixture quotes the supplied report excerpt. [{reference}]"
                    )
                else:
                    answers.append(
                        "Bhatla identifies lost or stolen cards, identity theft, "
                        "skimming, counterfeit cards and mail interception among fraud mechanisms. "
                        f"This offline fixture quotes the supplied report excerpt. [{reference}]"
                    )
                citations.append(reference)
            return schema.model_validate(
                {"answer": "\n\n".join(answers), "status": "answered", "citations": citations}
            )
        return schema.model_validate(
            {
                "evidence_support": 0.9,
                "relevance": 0.85,
                "completeness": 0.8,
                "consistency": 1.0,
                "explanation": "Demonstration rubric values from an offline fixture; no model evaluation was performed.",
            }
        )


class FixtureRetriever:
    def __init__(self, directory):
        self.corpus = load_corpus(directory)

    def retrieve(self, question, document_id=None, deadline=None):
        wanted = document_id or "eba"
        page = 4 if wanted == "bhatla" else 27
        candidates = [
            c for c in self.corpus.chunks if c["document_id"] == wanted and c["page"] == page
        ]
        # Select the excerpt containing the actual tabular/narrative gold facts.
        text = "71%" if wanted == "eba" else "48%"
        chunk = next((c for c in candidates if text in c["text"]), candidates[0])
        return [
            {
                "citation_id": f"{wanted}-p{page}-{chunk['chunk_id'][:8]}",
                "payload": chunk,
                "rerank_score": None,
            }
        ]


class FixtureResources:
    def __init__(self):
        self.settings = Settings(_env_file=None)
        self.llm = FixtureLLM()
        self.retriever = FixtureRetriever(self.settings.corpus_dir)

    def query(self, sql, deadline):
        return SimpleNamespace(
            sql=sql + " LIMIT 201",
            truncated=False,
            rows=[
                {"year_month": "2020-01", "total_tx": 10000, "fraud_tx": 200, "fraud_rate": 0.02},
                {"year_month": "2020-02", "total_tx": 10000, "fraud_tx": 100, "fraud_rate": 0.01},
                {"year_month": "2020-03", "total_tx": 10000, "fraud_tx": 150, "fraud_rate": 0.015},
            ],
        )

    def ready(self):
        return {
            "status": "ok",
            "db_ok": True,
            "qdrant_ok": True,
            "models_cached": False,
            "openai_configured": False,
            "model": "offline-fixture",
            "mode": "offline_fixture",
        }

    def close(self):
        pass


app = create_app(resources=FixtureResources(), settings=Settings(_env_file=None))
