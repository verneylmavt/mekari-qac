"""Strict warehouse SQL policy and bounded PostgreSQL read transactions."""

import base64
import math
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import sqlglot
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlglot import exp
from sqlglot.errors import ErrorLevel, OptimizeError, ParseError, TokenError
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.scope import Scope, traverse_scope

from ..db import get_engine

WAREHOUSE_TABLES = frozenset(
    {
        "fact_transactions",
        "dim_customer",
        "dim_merchant",
        "dim_category",
        "dim_date",
        "agg_daily_fraud",
        "agg_monthly_fraud",
        "agg_merchant_fraud",
        "agg_category_fraud",
    }
)
MAX_COLUMNS = 100
MAX_CELL_BYTES = 16_384
MAX_RESULT_BYTES = 1_000_000
MAX_SQL_CHARS = 20_000
MAX_AST_NODES = 2_000

# Explicit functions keep extensions and user-defined routines outside the policy.
SAFE_FUNCTIONS = frozenset(
    {
        "ABS",
        "AVG",
        "CEIL",
        "CEILING",
        "COALESCE",
        "COUNT",
        "DATE_TRUNC",
        "TIMESTAMP_TRUNC",
        "EXTRACT",
        "FLOOR",
        "GREATEST",
        "LEAST",
        "LOWER",
        "UPPER",
        "LENGTH",
        "CHAR_LENGTH",
        "MAX",
        "MIN",
        "NULLIF",
        "ROUND",
        "SUM",
        "STDDEV",
        "STDDEV_POP",
        "STDDEV_SAMP",
        "VARIANCE",
        "VAR_POP",
        "VAR_SAMP",
        "PERCENTILE_CONT",
        "PERCENTILE_DISC",
        "ROW_NUMBER",
        "RANK",
        "DENSE_RANK",
        "LAG",
        "LEAD",
        "FIRST_VALUE",
        "LAST_VALUE",
        "NTH_VALUE",
        "NTILE",
        "CUME_DIST",
        "PERCENT_RANK",
        "BOOL_AND",
        "BOOL_OR",
        "TO_CHAR",
        "TIME_TO_STR",
        "VARIANCE_POP",
        "CAST",
        "CASE",
        "IF",
        "LOGICAL_AND",
        "LOGICAL_OR",
    }
)
SAFE_NODES = frozenset(
    {
        "select",
        "union",
        "intersect",
        "except",
        "subquery",
        "with",
        "cte",
        "table",
        "tablealias",
        "from",
        "join",
        "where",
        "group",
        "having",
        "order",
        "ordered",
        "limit",
        "offset",
        "distinct",
        "filter",
        "window",
        "windowspec",
        "withingroup",
        "partition",
        "identifier",
        "column",
        "star",
        "alias",
        "literal",
        "null",
        "boolean",
        "var",
        "paren",
        "tuple",
        "datatype",
        "datatypeparam",
        "interval",
        "and",
        "or",
        "not",
        "eq",
        "neq",
        "gt",
        "gte",
        "lt",
        "lte",
        "is",
        "between",
        "in",
        "exists",
        "like",
        "ilike",
        "escape",
        "neg",
        "add",
        "sub",
        "mul",
        "div",
        "mod",
        "pow",
        "dpipe",
    }
)
SAFE_CAST_TYPES = frozenset(
    {
        "BIGINT",
        "INT",
        "SMALLINT",
        "TINYINT",
        "DECIMAL",
        "DOUBLE",
        "FLOAT",
        "REAL",
        "BOOLEAN",
        "TEXT",
        "VARCHAR",
        "CHAR",
        "DATE",
        "TIMESTAMP",
        "TIMESTAMPTZ",
        "TIME",
        "INTERVAL",
    }
)


class SQLPolicyError(ValueError):
    code = "sql_policy_violation"
    repairable = False


class SQLExecutionError(RuntimeError):
    def __init__(
        self, message: str, *, code: str = "sql_execution_failed", repairable: bool = False
    ):
        super().__init__(message)
        self.code = code
        self.repairable = repairable


@dataclass(frozen=True)
class SQLQueryResult:
    sql: str
    rows: list[dict[str, Any]]
    truncated: bool


def _integer_literal(node: exp.Expression | None) -> int:
    if not isinstance(node, exp.Literal) or node.is_string or not node.this.isdecimal():
        raise SQLPolicyError("SQL limits and offsets must be nonnegative integer literals.")
    return int(node.this)


def validate_sql(sql: str, *, max_rows: int = 200) -> str:
    """Parse one PostgreSQL query, resolve CTE scopes, and cap its outer limit."""
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= 200:
        raise ValueError("max_rows must be between 1 and 200.")
    if not isinstance(sql, str) or not sql.strip() or len(sql) > MAX_SQL_CHARS:
        raise SQLPolicyError("SQL is empty or exceeds the allowed query size.")
    try:
        statements = sqlglot.parse(sql, read="postgres", error_level=ErrorLevel.RAISE)
        statements = [
            statement
            for statement in statements
            if statement is not None and not isinstance(statement, exp.Semicolon)
        ]
        if len(statements) != 1 or not isinstance(
            statements[0], (exp.Select, exp.Union, exp.Intersect, exp.Except)
        ):
            raise SQLPolicyError("Only one read-only SELECT query is allowed.")
        query = normalize_identifiers(statements[0], dialect="postgres")
        nodes = list(query.walk())
        if len(nodes) > MAX_AST_NODES:
            raise SQLPolicyError("SQL exceeds the allowed query complexity.")
        for node in nodes:
            if isinstance(node, exp.Func):
                name = node.name if isinstance(node, exp.Anonymous) else node.sql_name()
                if name.upper() not in SAFE_FUNCTIONS:
                    raise SQLPolicyError("SQL uses a function outside the analytic allowlist.")
            elif node.key not in SAFE_NODES:
                raise SQLPolicyError("SQL uses a construct outside the read-only allowlist.")
            if isinstance(node, exp.With) and node.args.get("recursive"):
                raise SQLPolicyError("Recursive queries are not allowed.")
            if isinstance(node, exp.Select) and (node.args.get("into") or node.args.get("locks")):
                raise SQLPolicyError("SELECT INTO and row locking are not allowed.")
            if isinstance(node, exp.DataType) and node.this.value not in SAFE_CAST_TYPES:
                raise SQLPolicyError("SQL uses a type outside the analytic allowlist.")
            if isinstance(node, (exp.Limit, exp.Offset)):
                _integer_literal(node.expression)
        audited_tables = set()
        for scope in traverse_scope(query):
            for table in scope.tables:
                source = scope.sources.get(table.alias_or_name)
                if isinstance(source, Scope) and not table.db and not table.catalog:
                    audited_tables.add(id(table))
                    continue
                if (
                    not isinstance(table.this, exp.Identifier)
                    or table.catalog
                    or table.db not in ("", "public")
                    or table.name not in WAREHOUSE_TABLES
                ):
                    raise SQLPolicyError(
                        "SQL may only read the allowlisted public warehouse tables."
                    )
                table.set("db", exp.to_identifier("public"))
                audited_tables.add(id(table))
        if any(id(table) not in audited_tables for table in query.find_all(exp.Table)):
            raise SQLPolicyError("SQL contains an unresolved table source.")
        limit = query.args.get("limit")
        # One extra row tells the caller whether evidence has been truncated.
        row_limit = min(_integer_literal(limit.expression), max_rows + 1) if limit else max_rows + 1
        query.set("limit", exp.Limit(expression=exp.Literal.number(row_limit)))
        return query.sql(dialect="postgres", unsupported_level=ErrorLevel.RAISE)
    except (
        ParseError,
        TokenError,
        OptimizeError,
        sqlglot.errors.UnsupportedError,
        RecursionError,
    ) as error:
        raise SQLPolicyError("SQL could not be parsed as a supported PostgreSQL query.") from error


def _safe_database_error(error: SQLAlchemyError) -> SQLExecutionError:
    original = getattr(error, "orig", None)
    sqlstate = getattr(original, "pgcode", None) or getattr(original, "sqlstate", None)
    if sqlstate == "57014":
        return SQLExecutionError(
            "The database query exceeded its execution deadline.", code="sql_timeout"
        )
    if sqlstate in {"42601", "42703", "42P01", "42883", "42702", "42P02"}:
        return SQLExecutionError(
            "The query has a syntax or warehouse schema error.",
            code="sql_invalid_query",
            repairable=True,
        )
    return SQLExecutionError("The database query could not be completed.")


def _json_value(value):
    """Keep database evidence numeric and temporal types consistent on the wire."""
    if isinstance(value, Decimal):
        if not value.is_finite() or not math.isfinite(float(value)):
            raise SQLExecutionError("The query returned invalid numeric evidence.")
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise SQLExecutionError("The query returned invalid numeric evidence.")
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    return value


def run_sql_query(
    sql: str, *, engine: Engine | None = None, max_rows: int = 200, timeout_ms: int = 10000
) -> SQLQueryResult:
    safe_sql = validate_sql(sql, max_rows=max_rows)
    if (
        isinstance(timeout_ms, bool)
        or not isinstance(timeout_ms, int)
        or not 1 <= timeout_ms <= 10000
    ):
        raise ValueError("timeout_ms must be between 1 and 10000.")
    try:
        with (engine if engine is not None else get_engine()).connect() as conn:
            transaction = conn.begin()
            result = None
            try:
                conn.exec_driver_sql("SET TRANSACTION READ ONLY")
                conn.exec_driver_sql(f"SET LOCAL statement_timeout = {timeout_ms}")
                conn.exec_driver_sql("SET LOCAL search_path = pg_catalog")
                conn.exec_driver_sql("SET LOCAL standard_conforming_strings = on")
                result = conn.exec_driver_sql(safe_sql, execution_options={"no_parameters": True})
                if len(result.keys()) > MAX_COLUMNS:
                    raise SQLExecutionError(
                        "The query returned too many columns.", code="sql_result_too_large"
                    )
                rows = [dict(row) for row in result.mappings().fetchmany(max_rows + 1)]
                total_bytes = 0
                for row in rows:
                    for value in row.values():
                        cell_bytes = (
                            len(value)
                            if isinstance(value, bytes)
                            else len(str(value).encode("utf-8"))
                        )
                        total_bytes += cell_bytes
                        if cell_bytes > MAX_CELL_BYTES or total_bytes > MAX_RESULT_BYTES:
                            raise SQLExecutionError(
                                "The query returned oversized evidence.",
                                code="sql_result_too_large",
                            )
                normalized = [
                    {key: _json_value(value) for key, value in row.items()}
                    for row in rows[:max_rows]
                ]
                return SQLQueryResult(safe_sql, normalized, len(rows) > max_rows)
            finally:
                try:
                    if result is not None:
                        result.close()
                finally:
                    transaction.rollback()
    except SQLAlchemyError as error:
        raise _safe_database_error(error) from None
