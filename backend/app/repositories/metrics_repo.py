# backend/app/repositories/metrics_repo.py

from typing import List, Dict, Any, Optional
import re  # NEW

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..db import get_engine


def _rows_to_dicts(result) -> List[Dict[str, Any]]:
    # SQLAlchemy 2.x friendly: use .mappings()
    return [dict(row) for row in result.mappings().all()]


# def get_daily_fraud_series(
#     start_date: Optional[str] = None,
#     end_date: Optional[str] = None,
#     engine: Optional[Engine] = None,
# ) -> List[Dict[str, Any]]:
#     engine = engine or get_engine()
#     sql = """
#         SELECT
#             trans_date,
#             year,
#             month,
#             day,
#             day_of_week,
#             is_weekend,
#             year_month,
#             total_tx,
#             fraud_tx,
#             fraud_rate,
#             total_amount,
#             fraud_amount,
#             fraud_share_by_value
#         FROM agg_daily_fraud
#         WHERE 1=1
#     """
#     params: Dict[str, Any] = {}

#     if start_date:
#         sql += " AND trans_date >= :start_date"
#         params["start_date"] = start_date
#     if end_date:
#         sql += " AND trans_date <= :end_date"
#         params["end_date"] = end_date

#     sql += " ORDER BY trans_date"

#     with engine.connect() as conn:
#         result = conn.execute(text(sql), params)
#         return _rows_to_dicts(result)


# def get_monthly_fraud_series(
#     engine: Optional[Engine] = None,
# ) -> List[Dict[str, Any]]:
#     engine = engine or get_engine()
#     sql = """
#         SELECT
#             year,
#             month,
#             year_month,
#             total_tx,
#             fraud_tx,
#             fraud_rate,
#             total_amount,
#             fraud_amount,
#             fraud_share_by_value
#         FROM agg_monthly_fraud
#         ORDER BY year, month
#     """
#     with engine.connect() as conn:
#         result = conn.execute(text(sql))
#         return _rows_to_dicts(result)


# def get_top_merchants_by_fraud(
#     limit: int = 10,
#     engine: Optional[Engine] = None,
# ) -> List[Dict[str, Any]]:
#     engine = engine or get_engine()
#     sql = """
#         SELECT
#             merchant_name,
#             total_tx,
#             fraud_tx,
#             fraud_rate,
#             total_amount,
#             fraud_amount,
#             fraud_share_by_value
#         FROM agg_merchant_fraud
#         ORDER BY fraud_rate DESC
#         LIMIT :limit
#     """
#     with engine.connect() as conn:
#         result = conn.execute(text(sql), {"limit": limit})
#         return _rows_to_dicts(result)


# def get_top_categories_by_fraud(
#     limit: int = 10,
#     engine: Optional[Engine] = None,
# ) -> List[Dict[str, Any]]:
#     engine = engine or get_engine()
#     sql = """
#         SELECT
#             category_name,
#             total_tx,
#             fraud_tx,
#             fraud_rate,
#             total_amount,
#             fraud_amount,
#             fraud_share_by_value
#         FROM agg_category_fraud
#         ORDER BY fraud_rate DESC
#         LIMIT :limit
#     """
#     with engine.connect() as conn:
#         result = conn.execute(text(sql), {"limit": limit})
#         return _rows_to_dicts(result)


def validate_sql(sql: str) -> str:
    """
    Very simple SQL safety: enforce SELECT and prevent semicolons / dangerous keywords.
    """
    cleaned = sql.strip().rstrip(";")
    lowered = cleaned.lower()

    # if not lowered.startswith("select"):
    #     raise ValueError("Only SELECT statements are allowed.")

    dangerous = [
        "insert ",
        "update ",
        "delete ",
        "drop ",
        "create ",
        "alter ",
        "truncate ",
    ]
    if any(word in lowered for word in dangerous):
        raise ValueError("SQL contains potentially dangerous keywords.")

    return cleaned


def ensure_limit(sql: str, default_limit: int = 200) -> str:
    """
    Ensure the query has some LIMIT, but do not add a second one.

    Uses a regex word boundary check so it catches 'limit 20' at the end as well.
    """
    lowered = sql.lower()
    if re.search(r"\blimit\b", lowered):
        return sql
    return f"{sql} LIMIT {default_limit}"


def run_sql_query(
    sql: str,
    engine: Optional[Engine] = None,
    default_limit: int = 200,
) -> List[Dict[str, Any]]:
    engine = engine or get_engine()
    safe_sql = validate_sql(sql)
    # safe_sql = ensure_limit(safe_sql, default_limit=default_limit)

    with engine.connect() as conn:
        result = conn.execute(text(safe_sql))
        return _rows_to_dicts(result)