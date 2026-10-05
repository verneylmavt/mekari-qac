import json
from decimal import Decimal
from numbers import Real

from ..rdb.postgresql_client import SQLExecutionError, SQLPolicyError
from ..runtime import ServiceError
from .state import SQLDraft
from .warehouse_schema import SCHEMA_DESCRIPTION

SQL_SYSTEM = (
    """Generate one read-only PostgreSQL SELECT over the documented public warehouse.
Prefer agg_* views. Fraud rate is fraud transaction count / total transaction count; fraud value
share is fraud amount / total amount. These are fractions, not percentages. Never average rates:
overall rate is SUM(fraud_tx)::float / NULLIF(SUM(total_tx),0). Order temporal rows chronologically.
Rank incidence by fraud_tx unless a rate is explicitly requested; show total_tx alongside fraud_rate
and mention small denominators. Prefer aggregated statistics and exclude personal names, addresses,
card numbers, birthdates and transaction identifiers. No external relations/functions, comments,
locking or writes. Do not claim this synthetic dataset represents EBA EU/EEA statistics.
Use the schema only; return the SQL in the structured sql field.
"""
    + SCHEMA_DESCRIPTION
)


def evidence_summary(rows):
    """Weighted count/value rates only, never a generic average across identifiers or ratios."""
    result = {}
    for numerator, denominator, label in (
        ("fraud_tx", "total_tx", "fraud_rate"),
        ("fraud_amount", "total_amount", "fraud_share_by_value"),
    ):
        if rows and all(
            isinstance(r.get(numerator), (Real, Decimal))
            and isinstance(r.get(denominator), (Real, Decimal))
            and not isinstance(r.get(numerator), bool)
            and not isinstance(r.get(denominator), bool)
            for r in rows
        ):
            total = sum(Decimal(str(r[denominator])) for r in rows)
            if total:
                result[label] = float(sum(Decimal(str(r[numerator])) for r in rows) / total)
    return result


def run_sql_node(state):
    resources, deadline = state["resources"], state["deadline"]
    question = state["plan"].data_question
    draft = resources.llm.complete(
        "sql", SQLDraft, SQL_SYSTEM, json.dumps({"question": question}), deadline
    )
    for attempt in range(2):
        deadline.check()
        try:
            result = resources.query(draft.sql, deadline)
            deadline.check()
            state.update(sql=result.sql, rows=result.rows, truncated=result.truncated)
            return state
        except SQLPolicyError:
            state.update(
                rows=[],
                status="insufficient_evidence",
                answer="I could not produce a query that meets the warehouse read-only policy.",
            )
            return state
        except SQLExecutionError as exc:
            deadline.check()
            if attempt == 0 and exc.repairable:
                draft = resources.llm.complete(
                    "sql",
                    SQLDraft,
                    SQL_SYSTEM,
                    json.dumps(
                        {
                            "question": question,
                            "previous_sql": draft.sql,
                            "repair_reason": exc.code,
                            "instruction": "Repair the syntax/schema issue using only this schema.",
                        }
                    ),
                    deadline,
                )
                continue
            raise ServiceError(
                "warehouse_unavailable", "The warehouse query could not be completed."
            ) from None
    raise AssertionError("unreachable")


def sql_evidence(state):
    rows = state.get("rows", [])
    preview_limit = state["resources"].settings.sql_preview_rows
    return {
        "citation_id": "SQL",
        "scope": "Synthetic transaction warehouse",
        "sql": state.get("sql"),
        "rows_preview": rows[:preview_limit],
        "returned_rows": len(rows),
        "truncated": state.get("truncated", False),
        "preview_truncated": len(rows) > preview_limit,
        "weighted_rates_over_returned_rows": evidence_summary(rows),
        "units": "fraud_rate and fraud_share_by_value are fractions; amount currency unspecified",
    }
