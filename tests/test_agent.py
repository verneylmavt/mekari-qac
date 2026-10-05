import json
from types import SimpleNamespace

import pytest

from backend.app.config import Settings
from backend.app.schemas import ChatRequest


class FakeLLM:
    def __init__(self, plan, answer=None, judge=None):
        self.plan = plan
        self.answer = answer
        self.judge = judge
        self.calls = []

    def complete(self, task, schema, system, user, deadline):
        self.calls.append((task, system, json.loads(user)))
        if task == "planner":
            value = self.plan
        elif task == "sql":
            value = {
                "sql": "SELECT year_month, fraud_rate FROM agg_monthly_fraud ORDER BY year_month"
            }
        elif task == "answer":
            value = self.answer
        else:
            if isinstance(self.judge, Exception):
                raise self.judge
            value = self.judge or {
                "evidence_support": 0.9,
                "relevance": 0.9,
                "completeness": 0.8,
                "consistency": 1.0,
                "explanation": "Claims are supported by the supplied evidence.",
            }
        return schema.model_validate(value)


def plan(route="document", question="What share is cross-border?"):
    return {
        "route": route,
        "standalone_question": question,
        "document_id": "eba",
        "data_question": question if route in ("data", "mixed") else "",
        "document_question": question if route in ("document", "mixed") else "",
        "clarification": "",
    }


def chunk():
    return {
        "citation_id": "EBA-P27-C1",
        "payload": {
            "chunk_id": "id1",
            "document_id": "eba",
            "title": "EBA 2024 report",
            "page": 27,
            "section": "Card fraud",
            "text": "In H1 2023, 71% of card fraud value and 68% of its volume were cross-border.",
            "source_sha256": "a" * 64,
        },
        "rerank_score": 4.0,
    }


class FakeRetriever:
    def __init__(self, chunks=None):
        self.chunks = [chunk()] if chunks is None else chunks
        self.calls = []

    def retrieve(self, question, document_id=None, deadline=None):
        self.calls.append((question, document_id))
        return self.chunks


def resources(llm, retriever=None):
    return SimpleNamespace(
        settings=Settings(_env_file=None),
        llm=llm,
        retriever=retriever or FakeRetriever(),
        query=lambda sql, deadline: SimpleNamespace(
            sql=sql + " LIMIT 200",
            rows=[{"year_month": "2020-01", "fraud_rate": 0.01}],
            truncated=False,
        ),
    )


def run(request, resources):
    from backend.app.agent.state_graph import run_agent

    return run_agent(request, resources)


def test_history_resolution_and_exclusive_document_filter():
    llm = FakeLLM(
        plan(question="What share of card fraud value was cross-border in H1 2023?"),
        {
            "answer": "71% of card fraud value was cross-border. [EBA-P27-C1]",
            "status": "answered",
            "citations": ["EBA-P27-C1"],
        },
    )
    retriever = FakeRetriever()
    result = run(
        ChatRequest(
            question="And its value?",
            document_id="eba",
            history=[
                {"role": "user", "content": "EBA cross-border card fraud in H1 2023?"},
                {"role": "assistant", "content": "68% of volume was cross-border."},
            ],
        ),
        resources(llm, retriever),
    )
    assert result.answer_type == "document"
    assert result.quality_available
    assert llm.calls[0][2]["history"][0]["role"] == "user"
    assert retriever.calls == [(llm.plan["document_question"], "eba")]
    assert llm.calls[-1][2]["evidence"] == llm.calls[-2][2]["evidence"]


def test_unknown_citation_abstains_before_judge():
    llm = FakeLLM(
        plan(), {"answer": "71%. [INVENTED]", "status": "answered", "citations": ["INVENTED"]}
    )
    result = run(ChatRequest(question="Share?"), resources(llm))
    assert result.status == "insufficient_evidence"
    assert not result.quality_available
    assert "judge" not in [c[0] for c in llm.calls]


def test_unsupported_number_abstains():
    llm = FakeLLM(
        plan(),
        {"answer": "99% of value. [EBA-P27-C1]", "status": "answered", "citations": ["EBA-P27-C1"]},
    )
    result = run(ChatRequest(question="Share?"), resources(llm))
    assert result.status == "insufficient_evidence"


def test_empty_evidence_skips_answer_provider():
    llm = FakeLLM(plan())
    result = run(ChatRequest(question="Share?"), resources(llm, FakeRetriever([])))
    assert result.status == "insufficient_evidence"
    assert [c[0] for c in llm.calls] == ["planner"]


def test_mixed_answer_keeps_both_sources_and_scopes():
    llm = FakeLLM(
        plan("mixed"),
        {
            "answer": "Synthetic data: 1% fraud rate. [SQL]\n\nEBA: 71% of card fraud value. [EBA-P27-C1]",
            "status": "answered",
            "citations": ["SQL", "EBA-P27-C1"],
        },
    )
    result = run(ChatRequest(question="Compare our rate to EBA."), resources(llm))
    assert result.answer_type == "mixed"
    assert {s["type"] for s in result.sources} == {"sql_result", "document_chunks"}
    assert result.sql.endswith("LIMIT 200")
    assert "fraud_rate" in llm.calls[-2][2]["evidence"]


def test_judge_outage_is_reported_without_fake_confidence():
    from backend.app.runtime import ServiceError

    llm = FakeLLM(
        plan(),
        {"answer": "71% of value. [EBA-P27-C1]", "status": "answered", "citations": ["EBA-P27-C1"]},
        judge=ServiceError("provider", "Down"),
    )
    result = run(ChatRequest(question="Share?"), resources(llm))
    assert result.status == "answered"
    assert result.quality_score == 0
    assert not result.quality_available
    assert result.quality_method == "unavailable"


def test_clarification_skips_all_evidence_and_generation():
    value = plan("none")
    value["clarification"] = "Which document or metric do you mean?"
    llm = FakeLLM(value)
    result = run(ChatRequest(question="What about it?"), resources(llm))
    assert result.status == "clarification"
    assert [c[0] for c in llm.calls] == ["planner"]


def test_weighted_rate_uses_counts_not_mean_of_rates():
    from backend.app.agent.data_node import evidence_summary

    result = evidence_summary(
        [
            {"total_tx": 10, "fraud_tx": 2, "fraud_rate": 0.2},
            {"total_tx": 90, "fraud_tx": 0, "fraud_rate": 0},
        ]
    )
    assert result["fraud_rate"] == 0.02


def test_sql_schema_error_repaired_once_and_actual_sql_is_returned():
    from backend.app.rdb.postgresql_client import SQLExecutionError

    llm = FakeLLM(
        plan("data"),
        {"answer": "Synthetic rate is 1%. [SQL]", "status": "answered", "citations": ["SQL"]},
    )
    deps = resources(llm)
    calls = []

    def query(sql, deadline):
        calls.append(sql)
        if len(calls) == 1:
            raise SQLExecutionError("Safe", code="sql_invalid_query", repairable=True)
        return SimpleNamespace(
            sql="SELECT fraud_rate FROM public.agg_monthly_fraud LIMIT 201",
            rows=[{"fraud_rate": 0.01}],
            truncated=False,
        )

    deps.query = query
    result = run(ChatRequest(question="Rate?"), deps)
    assert len(calls) == 2
    assert [c[0] for c in llm.calls].count("sql") == 2
    assert "public.agg_monthly_fraud" in result.sql


def test_sql_policy_violation_is_never_repaired_or_answered():
    from backend.app.rdb.postgresql_client import SQLPolicyError

    llm = FakeLLM(plan("data"))
    deps = resources(llm)

    def query(sql, deadline):
        raise SQLPolicyError("Only warehouse tables")

    deps.query = query
    result = run(ChatRequest(question="Rate?"), deps)
    assert result.status == "insufficient_evidence"
    assert [c[0] for c in llm.calls] == ["planner", "sql"]


def test_low_support_quality_rejects_answer():
    llm = FakeLLM(
        plan(),
        {"answer": "71% of value. [EBA-P27-C1]", "status": "answered", "citations": ["EBA-P27-C1"]},
        judge={
            "evidence_support": 0.1,
            "relevance": 1,
            "completeness": 1,
            "consistency": 1,
            "explanation": "Unsupported interpretation.",
        },
    )
    result = run(ChatRequest(question="Share?"), resources(llm))
    assert result.status == "insufficient_evidence"
    assert result.quality_method == "rejected"


def test_full_chunks_not_sliced_and_sql_prompt_preview_capped():
    llm = FakeLLM(
        plan("mixed"),
        {
            "answer": "1% synthetic rate. [SQL]\n\n71% EBA value. [EBA-P27-C1]",
            "status": "answered",
            "citations": ["SQL", "EBA-P27-C1"],
        },
    )
    source = chunk()
    source["payload"]["text"] += "\n" + "evidence " * 120
    deps = resources(llm, FakeRetriever([source]))
    deps.query = lambda s, d: SimpleNamespace(
        sql=s, rows=[{"fraud_rate": 0.01}] * 200, truncated=True
    )
    result = run(ChatRequest(question="Compare."), deps)
    evidence = json.loads(llm.calls[-2][2]["evidence"])
    assert len(evidence["warehouse"]["rows_preview"]) == 20
    assert len(evidence["documents"][0]["text"]) > 1000
    assert result.truncated


def test_numbers_must_belong_to_paragraphs_cited_sources():
    from backend.app.agent.doc_node import numbers_supported

    assert not numbers_supported(
        "Synthetic fraud rate is 71%. [SQL]", [chunk()], [{"fraud_rate": 0.01}], {}
    )
    assert numbers_supported(
        "1% synthetic rate. [SQL]\n\n71% EBA value. [EBA-P27-C1]",
        [chunk()],
        [{"fraud_rate": 0.01}],
        {},
    )


def test_insufficient_draft_never_returns_unsupported_provider_claim():
    from backend.app.agent.doc_node import rag_answer_node
    from backend.app.agent.state import Plan
    from backend.app.runtime import Deadline

    llm = FakeLLM(
        plan(),
        {"answer": "99% was cross-border", "status": "insufficient_evidence", "citations": []},
    )
    state = {
        "resources": resources(llm),
        "deadline": Deadline(3),
        "plan": Plan.model_validate(plan()),
        "chunks": [chunk()],
    }
    result = rag_answer_node(state)
    assert "99%" not in result["answer"]


def test_expired_sql_deadline_is_504_even_when_driver_reports_timeout():
    from backend.app.agent.data_node import run_sql_node
    from backend.app.agent.state import Plan
    from backend.app.rdb.postgresql_client import SQLExecutionError
    from backend.app.runtime import Deadline, ServiceError

    now = [0.0]
    deadline = Deadline(1, clock=lambda: now[0])
    deps = resources(FakeLLM(plan("data")))

    def query(sql, deadline):
        now[0] = 2
        raise SQLExecutionError("Timeout", code="sql_timeout")

    deps.query = query
    with pytest.raises(ServiceError) as exc:
        run_sql_node(
            {"resources": deps, "deadline": deadline, "plan": Plan.model_validate(plan("data"))}
        )
    assert exc.value.status_code == 504


def test_daily_answer_accepts_date_and_datetime_from_sql_driver():
    from datetime import date, datetime

    from backend.app.agent.doc_node import numbers_supported

    for value in (date(2020, 1, 15), datetime(2020, 1, 15, 13, 45)):
        assert numbers_supported(
            "On 2020-01-15, fraud rate was 1%. [SQL]",
            [],
            [{"trans_date": value, "fraud_rate": 0.01}],
            {},
        )


def test_weighted_summary_accepts_native_postgresql_numeric_counts():
    from decimal import Decimal

    from backend.app.agent.data_node import evidence_summary

    assert evidence_summary(
        [
            {"fraud_tx": Decimal("1"), "total_tx": Decimal("100")},
            {"fraud_tx": Decimal("9"), "total_tx": Decimal("100")},
        ]
    ) == {"fraud_rate": 0.05}
