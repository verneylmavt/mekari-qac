"""Pure conversation and evidence transforms shared by UI and tests."""

from typing import Any
from uuid import uuid4

from frontend.client import finite_number


def build_history_for_backend(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    failed_ids = {
        message.get("turn_id")
        for message in messages
        if message.get("state") in ("failed", "pending") and message.get("turn_id")
    }
    history = []
    for message in messages:
        if message.get("state", "completed") != "completed" or message.get("turn_id") in failed_ids:
            continue
        role, content = message.get("role"), message.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            history.append({"role": role, "content": content[:8000]})
    history = history[-12:]
    if history and history[0]["role"] == "assistant":
        history = history[1:]
    return history


def begin_turn(
    messages: list[dict[str, Any]], question: str, document_id: str | None
) -> dict[str, Any]:
    turn = {
        "id": str(uuid4()),
        "question": question.strip(),
        "document_id": document_id,
        "history": build_history_for_backend(messages),
        "state": "pending",
    }
    messages.append(
        {
            "role": "user",
            "content": turn["question"],
            "turn_id": turn["id"],
            "state": "pending",
            "document_id": document_id,
        }
    )
    return turn


def retry_turn(messages: list[dict[str, Any]], turn: dict[str, Any]) -> dict[str, Any]:
    if turn.get("state") != "failed":
        raise ValueError("Only a failed turn can be retried.")
    retry = dict(turn, state="pending")
    for message in messages:
        if message.get("turn_id") == turn["id"] and message.get("role") == "user":
            message["state"] = "pending"
    retry.pop("error", None)
    return retry


def citation_label(chunk: dict[str, Any], index: int) -> str:
    payload = chunk.get("payload") if isinstance(chunk.get("payload"), dict) else chunk
    reference = chunk.get("citation_id") or payload.get("citation_id") or f"Source {index}"
    title = payload.get("title") or payload.get("section") or "Document evidence"
    page = payload.get("page")
    suffix = f" · page {page}" if page is not None else " · page unavailable"
    return f"[{reference}] {title}{suffix}"


def chart_spec(sources: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    dimensions = [
        ("year_month", "Month", "line"),
        ("category_name", "Category", "bar"),
        ("merchant_name", "Merchant", "bar"),
        ("category", "Category", "bar"),
        ("merchant", "Merchant", "bar"),
    ]
    measures = [
        ("fraud_rate", "Fraud rate (%)", 100),
        ("fraud_share_by_value", "Fraud share by value (%)", 100),
        ("fraud_tx", "Fraud transactions (count)", 1),
        ("total_tx", "Transactions (count)", 1),
        ("fraud_count", "Fraud transactions (count)", 1),
        ("fraud_txn", "Fraud transactions (count)", 1),
        ("txn_count", "Transactions (count)", 1),
        ("n_txn", "Transactions (count)", 1),
        ("fraud_amount", "Fraud amount (dataset units)", 1),
        ("fraud_amt", "Fraud amount (dataset units)", 1),
        ("total_amount", "Transaction amount (dataset units)", 1),
    ]
    for source in sources or []:
        if source.get("type") != "sql_result":
            continue
        rows = source.get("rows_preview") or []
        if not rows:
            continue
        for dimension, x, kind in dimensions:
            for measure, y, scale in measures:
                if not all(
                    dimension in row and finite_number(row.get(measure)) is not None for row in rows
                ):
                    continue
                if scale == 100 and not all(0 <= row[measure] <= 1 for row in rows):
                    continue
                plotted = [{x: str(row[dimension]), y: row[measure] * scale} for row in rows]
                if kind == "line":
                    plotted.sort(key=lambda row: row[x])
                return {"rows": plotted, "x": x, "y": y, "kind": kind}
    return None


def export_markdown(messages: list[dict[str, Any]]) -> str:
    sections = [
        "# Fraud intelligence conversation",
        "Evidence is limited to returned SQL rows and cited document excerpts.",
    ]
    for message in messages:
        if message.get("state", "completed") != "completed":
            continue
        sections.extend(
            [
                f"## {'Question' if message.get('role') == 'user' else 'Answer'}",
                str(message.get("content", "")),
            ]
        )
        if message.get("role") != "assistant":
            continue
        score = finite_number(message.get("quality_score"))
        sections.append(
            f"Quality: {score:.0%}"
            if message.get("quality_available", score is not None) and score is not None
            else "Quality: N/A"
        )
        if message.get("sql"):
            sections.append(f"```sql\n{message['sql']}\n```")
        for source in message.get("sources") or []:
            if source.get("type") == "document_chunks":
                for index, chunk in enumerate(source.get("chunks") or [], 1):
                    sections.append(citation_label(chunk, index))
            elif source.get("type") == "sql_result":
                sections.append(
                    f"SQL preview: {len(source.get('rows_preview') or [])} rows; truncated: {bool(source.get('truncated', False))}."
                )
    return "\n\n".join(sections) + "\n"
