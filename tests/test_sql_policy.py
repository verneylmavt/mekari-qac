"""Offline boundary tests: queries must fail before opening a connection."""

import pytest

from backend.app.rdb import postgresql_client as sql_client


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; SELECT 2",
        "WITH x AS (DELETE FROM fact_transactions RETURNING *) SELECT * FROM x",
        "WITH RECURSIVE x AS (SELECT 1 UNION ALL SELECT 1 FROM x) SELECT * FROM x",
        "SELECT * INTO copied FROM fact_transactions",
        "SELECT * FROM fact_transactions FOR UPDATE",
        "SELECT * FROM pg_catalog.pg_roles",
        "SELECT * FROM private.fact_transactions",
        "SELECT * FROM arbitrary_table",
        "SELECT pg_sleep(100)",
        "SELECT public.sum(amt) FROM fact_transactions",
        "SELECT set_config('search_path', 'evil', false)",
        "SELECT * FROM generate_series(1, 1000000)",
        "WITH good AS (SELECT * FROM pg_roles) SELECT * FROM good",
        "WITH x AS (SELECT * FROM dim_date) SELECT * FROM x UNION SELECT * FROM pg_roles",
        "SELECT * FROM fact_transactions LIMIT ALL",
        "SELECT * FROM fact_transactions LIMIT (SELECT 2)",
        "SELECT 1 garbage garbage",
        "SELECT * FROM dim_date WHERE",
        "SELECT 'unterminated",
        "SELECT * FROM dim_date AS pg_roles JOIN pg_roles ON true",
        "WITH dim_date AS (SELECT * FROM pg_roles) SELECT * FROM dim_date",
        "SELECT CAST(1 AS regclass)",
    ],
)
def test_rejects_hostile_sql(sql):
    with pytest.raises(ValueError):
        sql_client.validate_sql(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) FROM public.fact_transactions",
        "WITH x AS (SELECT year_month, fraud_rate FROM agg_monthly_fraud) SELECT * FROM x",
        "SELECT coalesce(sum(amt) / nullif(count(*), 0), 0) FROM fact_transactions",
        "SELECT extract(year FROM trans_ts), date_trunc('month', trans_ts) FROM fact_transactions",
        "SELECT year_month, lag(fraud_rate) OVER (ORDER BY year_month) FROM agg_monthly_fraud",
        "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY amt) FROM fact_transactions",
        "SELECT to_char(trans_ts, 'YYYY-MM'), var_pop(amt) FROM fact_transactions GROUP BY 1",
        "SELECT 'drop table; -- comment' AS note FROM dim_date",
        "SELECT * FROM dim_date; -- harmless trailing comment",
        "SELECT count(*) FROM dim_date UNION ALL SELECT count(*) FROM dim_customer",
        "SELECT CASE WHEN amt > 100 THEN 'large' ELSE 'small' END, BOOL_OR(is_fraud = 1) FROM fact_transactions GROUP BY 1",
        "SELECT SUM(amt) FILTER (WHERE is_fraud=1), COUNT(DISTINCT merchant_id) FROM fact_transactions",
        "SELECT CAST(SUM(amt) AS numeric(10,2)) FROM fact_transactions",
        "SELECT year_month, SUM(total_tx) OVER (ORDER BY year_month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM agg_monthly_fraud",
    ],
)
def test_allows_warehouse_analytics_and_bounds_outer_query(sql):
    validated = sql_client.validate_sql(sql)
    assert "LIMIT 201" in validated.upper()


def test_preserves_smaller_outer_limit_but_caps_unbounded_union():
    assert "LIMIT 5" in sql_client.validate_sql("SELECT * FROM dim_date LIMIT 5").upper()
    validated = sql_client.validate_sql("SELECT * FROM dim_date LIMIT 999999")
    assert "LIMIT 201" in validated.upper()


class FakeResult:
    def __init__(self, rows, columns=None):
        self.rows = rows
        self.columns = columns or list(rows[0] if rows else {"value": 0})
        self.closed = False
        self.requested_rows = None

    def keys(self):
        return self.columns

    def mappings(self):
        return self

    def fetchmany(self, count):
        self.requested_rows = count
        return self.rows[:count]

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, result, error=None):
        self.result = result
        self.error = error
        self.commands = []
        self.rolled_back = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def begin(self):
        return self

    def rollback(self):
        self.rolled_back = True

    def exec_driver_sql(self, sql, execution_options=None):
        self.commands.append(sql)
        if sql.startswith("SET "):
            return FakeResult([])
        assert execution_options == {"no_parameters": True}
        if self.error:
            raise self.error
        return self.result

    def execute(self, sql):
        self.commands.append(str(sql))
        if self.error:
            raise self.error
        return self.result


class FakeEngine:
    def __init__(self, conn):
        self.conn = conn

    def connect(self):
        return self.conn


def test_transaction_is_readonly_bounded_and_always_rolled_back():
    result = FakeResult([{"value": index} for index in range(8)])
    conn = FakeConnection(result)
    response = sql_client.run_sql_query(
        "SELECT * FROM dim_date", engine=FakeEngine(conn), max_rows=3, timeout_ms=42
    )
    assert response.rows == [{"value": 0}, {"value": 1}, {"value": 2}]
    assert response.truncated is True
    assert "LIMIT 4" in response.sql
    assert conn.commands[:3] == [
        "SET TRANSACTION READ ONLY",
        "SET LOCAL statement_timeout = 42",
        "SET LOCAL search_path = pg_catalog",
    ]
    assert result.requested_rows == 4
    assert conn.rolled_back and result.closed


def test_policy_rejection_happens_before_connection_acquisition():
    class ForbiddenEngine:
        def connect(self):
            pytest.fail("A rejected query opened a database connection")

    with pytest.raises(sql_client.SQLPolicyError):
        sql_client.run_sql_query("SELECT pg_sleep(10)", engine=ForbiddenEngine())


def test_small_result_and_smaller_limit_report_complete_evidence():
    conn = FakeConnection(FakeResult([{"value": 1}]))
    response = sql_client.run_sql_query("SELECT * FROM dim_date LIMIT 1", engine=FakeEngine(conn))
    assert response.rows == [{"value": 1}]
    assert not response.truncated
    assert "LIMIT 1" in response.sql


def test_sql_execution_uses_standard_string_semantics_for_serialized_ast():
    conn = FakeConnection(FakeResult([{"note": "literal"}]))
    sql_client.run_sql_query(
        "SELECT 'backslash \\\\ and semicolon ;' AS note", engine=FakeEngine(conn)
    )
    assert "SET LOCAL standard_conforming_strings = on" in conn.commands[:-1]


def test_many_small_cells_cannot_exceed_total_evidence_budget():
    conn = FakeConnection(FakeResult([{"value": "x" * 10000} for _ in range(200)]))
    with pytest.raises(sql_client.SQLExecutionError) as exc:
        sql_client.run_sql_query("SELECT * FROM dim_date", engine=FakeEngine(conn))
    assert exc.value.code == "sql_result_too_large"
    assert conn.rolled_back


def test_rolls_back_even_when_cursor_cleanup_fails():
    from sqlalchemy.exc import SQLAlchemyError

    class BrokenClose(FakeResult):
        def close(self):
            raise SQLAlchemyError("driver cleanup failed")

    conn = FakeConnection(BrokenClose([]))
    with pytest.raises(sql_client.SQLExecutionError):
        sql_client.run_sql_query("SELECT * FROM dim_date", engine=FakeEngine(conn))
    assert conn.rolled_back


@pytest.mark.parametrize(
    "rows, columns", [([{"value": "x" * 50000}], None), ([], [f"c{x}" for x in range(101)])]
)
def test_rejects_oversized_cells_and_column_count(rows, columns):
    conn = FakeConnection(FakeResult(rows, columns))
    with pytest.raises(sql_client.SQLExecutionError) as exc:
        sql_client.run_sql_query("SELECT * FROM dim_date", engine=FakeEngine(conn))
    assert exc.value.code == "sql_result_too_large"
    assert not exc.value.repairable
    assert conn.rolled_back


@pytest.mark.parametrize(
    "state, repairable",
    [
        ("42601", True),
        ("42703", True),
        ("42P01", True),
        ("42501", False),
        ("57014", False),
        ("08006", False),
    ],
)
def test_errors_are_sanitized_and_repairability_is_narrow(state, repairable):
    from sqlalchemy.exc import DBAPIError

    class DriverError(Exception):
        pgcode = state

    error = DBAPIError("SECRET SQL", {}, DriverError("PASSWORD credentials"), False)
    conn = FakeConnection(FakeResult([]), error)
    with pytest.raises(sql_client.SQLExecutionError) as exc:
        sql_client.run_sql_query("SELECT * FROM dim_date", engine=FakeEngine(conn))
    assert exc.value.repairable is repairable
    assert "SECRET" not in str(exc.value) and "PASSWORD" not in str(exc.value)
    assert conn.rolled_back


def test_sql_temporal_and_numeric_values_have_consistent_json_types():
    from datetime import date, datetime
    from decimal import Decimal

    rows = [
        {
            "day": date(2020, 1, 15),
            "time": datetime(2020, 1, 15, 13, 45),
            "count": Decimal("3"),
            "ratio": Decimal("0.03"),
        }
    ]
    conn = FakeConnection(FakeResult(rows))
    result = sql_client.run_sql_query("SELECT * FROM dim_date", engine=FakeEngine(conn))
    assert result.rows == [
        {"day": "2020-01-15", "time": "2020-01-15T13:45:00", "count": 3, "ratio": 0.03}
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_sql_numbers_fail_safely_and_rollback(value):
    conn = FakeConnection(FakeResult([{"value": value}]))
    with pytest.raises(sql_client.SQLExecutionError):
        sql_client.run_sql_query("SELECT * FROM dim_date", engine=FakeEngine(conn))
    assert conn.rolled_back
