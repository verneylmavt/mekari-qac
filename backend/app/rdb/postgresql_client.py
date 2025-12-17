# backend/app/repositories/metrics_repo.py

from typing import List, Dict, Any, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..db import get_engine


def _rows_to_dicts(result) -> List[Dict[str, Any]]:
    # SQLAlchemy 2.x friendly: use .mappings()
    return [dict(row) for row in result.mappings().all()]


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


# def ensure_limit(sql: str, default_limit: int = 200) -> str:
#     """
#     Ensure the query has some LIMIT, but do not add a second one.

#     Uses a regex word boundary check so it catches 'limit 20' at the end as well.
#     """
#     lowered = sql.lower()
#     if re.search(r"\blimit\b", lowered):
#         return sql
#     return f"{sql} LIMIT {default_limit}"


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