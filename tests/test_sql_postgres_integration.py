"""Opt-in tests on the disposable local PostgreSQL fixture (zero external calls)."""

import os
from uuid import uuid4

import psycopg2
import pytest
from psycopg2 import sql
from sqlalchemy import URL, create_engine

from backend.app.rdb.postgresql_client import SQLExecutionError, run_sql_query
from scripts import init_postgresql as initializer

pytestmark = pytest.mark.skipif(
    os.getenv("FRAUD_TEST_LOCAL_POSTGRES") != "1",
    reason="requires explicitly enabled disposable PostgreSQL on 127.0.0.1:15432",
)


@pytest.fixture
def database():
    # This fixture intentionally cannot be pointed at an external/configured DB.
    identifier = uuid4().hex
    database_name = "fraud_test_" + identifier
    reader = "reader_" + identifier
    admin = psycopg2.connect(host="127.0.0.1", port=15432, user="postgres", dbname="postgres")
    admin.autocommit = True
    with admin.cursor() as cursor:
        cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
    config = initializer.RestoreConfig(
        "127.0.0.1", 15432, database_name, "postgres", "unused", reader, "unused", "unused"
    )
    connection = psycopg2.connect(
        host=config.host, port=config.port, user=config.admin_user, dbname=config.database
    )
    connection.autocommit = True
    try:
        yield config, connection
    finally:
        connection.close()
        with admin.cursor() as cursor:
            cursor.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database_name))
            )
            cursor.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(reader)))
        admin.close()


def warehouse(connection):
    with connection.cursor() as cursor:
        for name in initializer.WAREHOUSE_OBJECTS:
            cursor.execute(
                sql.SQL("CREATE TABLE public.{} (value INTEGER)").format(sql.Identifier(name))
            )
            cursor.execute(sql.SQL("INSERT INTO public.{} VALUES (1)").format(sql.Identifier(name)))


def reader_engine(config):
    return create_engine(
        URL.create(
            "postgresql+psycopg2",
            username=config.app_user,
            password=config.app_password,
            host=config.host,
            port=config.port,
            database=config.database,
        )
    )


def test_raw_query_preserves_colon_and_percent_literals(database):
    config, connection = database
    warehouse(connection)
    initializer.provision_reader(config)
    engine = reader_engine(config)
    try:
        result = run_sql_query("SELECT ':foo 100% literal' AS note", engine=engine)
        assert result.rows == [{"note": ":foo 100% literal"}]
    finally:
        engine.dispose()


def test_native_numeric_aggregates_survive_response_and_chart(database):
    from backend.app.agent.data_node import evidence_summary
    from backend.app.schemas import ChatResponse
    from frontend.presentation import chart_spec

    config, connection = database
    warehouse(connection)
    initializer.provision_reader(config)
    engine = reader_engine(config)
    try:
        result = run_sql_query(
            "SELECT '2020-01' AS year_month, SUM(value::bigint) AS fraud_tx, "
            "SUM((value * 100)::bigint) AS total_tx FROM dim_date",
            engine=engine,
        )
        summary = evidence_summary(result.rows)
        assert summary == {"fraud_rate": 0.01}
        response = ChatResponse(
            answer="One transaction. [SQL]",
            answer_type="data",
            quality_score=0,
            sources=[{"type": "sql_result", "rows_preview": result.rows}],
        ).model_dump(mode="json")
        assert response["sources"][0]["rows_preview"][0]["fraud_tx"] == 1
        chart = chart_spec(response["sources"])
        assert chart["rows"] == [{"Month": "2020-01", "Fraud transactions (count)": 1}]
    finally:
        engine.dispose()


def test_public_function_overload_cannot_override_builtin_resolution(database):
    config, connection = database
    warehouse(connection)
    with connection.cursor() as cursor:
        cursor.execute(
            "CREATE FUNCTION public.abs(TEXT) RETURNS INTEGER LANGUAGE SQL IMMUTABLE AS 'SELECT 43'"
        )
    initializer.provision_reader(config)
    engine = reader_engine(config)
    try:
        with pytest.raises(SQLExecutionError):
            run_sql_query("SELECT abs(CAST('trigger' AS TEXT))", engine=engine)
        with engine.connect() as reader_connection:
            assert (
                reader_connection.exec_driver_sql("SHOW search_path").scalar_one() == "pg_catalog"
            )
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "grant", ["SELECT", "INSERT", "UPDATE", "SELECT (secret)", "UPDATE (secret)"]
)
@pytest.mark.parametrize("schema", ["public", "private", "pgdata"])
def test_incompatible_public_table_or_column_grants_fail_before_restore(database, schema, grant):
    config, connection = database
    with connection.cursor() as cursor:
        if schema != "public":
            cursor.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        cursor.execute(
            sql.SQL("CREATE TABLE {}.other_table (secret TEXT)").format(sql.Identifier(schema))
        )
        cursor.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO PUBLIC").format(sql.Identifier(schema)))
        cursor.execute(
            sql.SQL("GRANT " + grant + " ON {}.other_table TO PUBLIC").format(
                sql.Identifier(schema)
            )
        )
    with pytest.raises(ValueError, match="privileges|grants"):
        initializer._check_target(config, replace=True)


def test_pgdata_schema_counts_as_nonempty(database):
    config, connection = database
    with connection.cursor() as cursor:
        cursor.execute("CREATE SCHEMA pgdata")
        cursor.execute("CREATE TABLE pgdata.existing (value INTEGER)")
    with pytest.raises(ValueError, match="--replace"):
        initializer._check_target(config, replace=False)


def test_existing_reader_column_grant_in_pgdata_is_rejected(database):
    config, connection = database
    with connection.cursor() as cursor:
        cursor.execute(sql.SQL("CREATE ROLE {} LOGIN").format(sql.Identifier(config.app_user)))
        cursor.execute("CREATE SCHEMA pgdata")
        cursor.execute("CREATE TABLE pgdata.existing (value INTEGER)")
        cursor.execute(
            sql.SQL("GRANT USAGE ON SCHEMA pgdata TO {}").format(sql.Identifier(config.app_user))
        )
        cursor.execute(
            sql.SQL("GRANT SELECT (value) ON pgdata.existing TO {}").format(
                sql.Identifier(config.app_user)
            )
        )
    with pytest.raises(ValueError, match="privileges|grants"):
        initializer._check_target(config, replace=True)


def test_reader_column_write_on_warehouse_is_rejected_before_restore(database):
    config, connection = database
    warehouse(connection)
    with connection.cursor() as cursor:
        cursor.execute(sql.SQL("CREATE ROLE {} LOGIN").format(sql.Identifier(config.app_user)))
        cursor.execute(
            sql.SQL("GRANT UPDATE (value) ON public.dim_date TO {}").format(
                sql.Identifier(config.app_user)
            )
        )
    with pytest.raises(ValueError, match="privileges|grants"):
        initializer._check_target(config, replace=True)


@pytest.mark.parametrize("reader", ["PUBLIC", "dedicated"])
def test_future_default_table_grants_are_rejected_before_restore(database, reader):
    config, connection = database
    with connection.cursor() as cursor:
        cursor.execute(sql.SQL("CREATE ROLE {} LOGIN").format(sql.Identifier(config.app_user)))
        grant_target = sql.SQL("PUBLIC") if reader == "PUBLIC" else sql.Identifier(config.app_user)
        cursor.execute(
            sql.SQL(
                "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {}"
            ).format(grant_target)
        )
    with pytest.raises(ValueError, match="privileges|grants"):
        initializer._check_target(config, replace=True)


def test_provisioned_reader_selects_warehouse_and_has_no_write_or_creation_grants(database):
    config, connection = database
    warehouse(connection)
    with connection.cursor() as cursor:
        cursor.execute("CREATE TABLE public.other_table (value INTEGER)")
    initializer.provision_reader(config)
    # Repeated initialization accepts the dedicated role's own warehouse grants.
    initializer._check_target(config, replace=True)
    reader = psycopg2.connect(
        host=config.host, port=config.port, user=config.app_user, dbname=config.database
    )
    reader.autocommit = True
    try:
        with reader.cursor() as cursor:
            cursor.execute("SET default_transaction_read_only = off")
            for name in initializer.WAREHOUSE_OBJECTS:
                cursor.execute(sql.SQL("SELECT * FROM public.{}").format(sql.Identifier(name)))
                assert cursor.fetchall() == [(1,)]
            for query in [
                "INSERT INTO public.dim_date VALUES (2)",
                "SELECT * FROM public.other_table",
                "CREATE TABLE public.denied (value INTEGER)",
                "CREATE TEMP TABLE denied (value INTEGER)",
            ]:
                with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                    cursor.execute(query)
    finally:
        reader.close()


def test_real_query_timeout_rolls_back_and_pool_can_run_next_query(database):
    config, connection = database
    warehouse(connection)
    with connection.cursor() as cursor:
        cursor.execute("INSERT INTO public.fact_transactions SELECT generate_series(1, 4000)")
    initializer.provision_reader(config)
    engine = reader_engine(config)
    try:
        with pytest.raises(SQLExecutionError) as error:
            run_sql_query(
                "SELECT count(*) FROM fact_transactions a CROSS JOIN fact_transactions b",
                engine=engine,
                timeout_ms=1,
            )
        assert error.value.code == "sql_timeout"
        assert not error.value.repairable
        assert run_sql_query("SELECT count(*) AS count FROM dim_date", engine=engine).rows == [
            {"count": 1}
        ]
        with engine.connect() as reader_connection:
            assert reader_connection.exec_driver_sql("SHOW statement_timeout").scalar_one() == "10s"
    finally:
        engine.dispose()
