"""Restore a validated snapshot into an already running PostgreSQL service."""

import argparse
import os
import shutil
import subprocess
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE_OBJECTS = (
    "fact_transactions",
    "dim_customer",
    "dim_merchant",
    "dim_category",
    "dim_date",
    "agg_daily_fraud",
    "agg_monthly_fraud",
    "agg_merchant_fraud",
    "agg_category_fraud",
)


@dataclass(frozen=True)
class RestoreConfig:
    host: str
    port: int
    database: str
    admin_user: str
    admin_password: str = field(repr=False)
    app_user: str
    app_password: str = field(repr=False)
    container: str
    container_port: int = 5432

    def validate(self):
        if self.app_user == self.admin_user:
            raise ValueError("Application and administrator roles must be separate.")
        if self.app_user.lower().startswith("pg_"):
            raise ValueError("Application roles must not use PostgreSQL's reserved pg_ prefix.")
        if not self.admin_password or not self.app_password:
            raise ValueError("DB_ADMIN_PASSWORD and DB_PASSWORD must both be configured.")
        if not all((self.host, self.database, self.admin_user, self.app_user, self.container)):
            raise ValueError("Database connection settings must not be empty.")
        if not 1 <= self.port <= 65535 or not 1 <= self.container_port <= 65535:
            raise ValueError("Database ports must be between 1 and 65535.")

    @classmethod
    def from_environment(cls):
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "database"),
            admin_user=os.getenv("DB_ADMIN_USER", "postgres"),
            admin_password=os.getenv("DB_ADMIN_PASSWORD", ""),
            app_user=os.getenv("DB_USER", "fraud_reader"),
            app_password=os.getenv("DB_PASSWORD", ""),
            container=os.getenv("DB_CONTAINER", "postgresql"),
            container_port=int(os.getenv("DB_CONTAINER_PORT", "5432")),
        )


def validate_snapshot(snapshot: Path) -> Path:
    snapshot = snapshot.resolve(strict=True)
    with snapshot.open("rb") as file:
        if file.read(5) != b"PGDMP":
            raise ValueError(
                "Snapshot must be a downloaded PostgreSQL PGDMP archive, not an LFS pointer."
            )
    return snapshot


def restore_command(
    config: RestoreConfig, *, executable: str | None, replace: bool
) -> tuple[list[str], dict[str, str]]:
    env = dict(os.environ, PGPASSWORD=config.admin_password, PGCONNECT_TIMEOUT="10")
    if executable:
        command = [executable]
        host, port = config.host, config.port
    else:
        command = [
            "docker",
            "exec",
            "-i",
            "-e",
            "PGPASSWORD",
            "-e",
            "PGCONNECT_TIMEOUT",
            config.container,
            "pg_restore",
        ]
        # Published host ports differ from the configured container's internal port.
        local_host = config.host in {"localhost", "127.0.0.1", "::1"}
        host = "localhost" if local_host else config.host
        port = config.container_port if local_host else config.port
    command.extend(
        [
            "--host",
            host,
            "--port",
            str(port),
            "--username",
            config.admin_user,
            "--dbname",
            config.database,
            "--single-transaction",
            "--exit-on-error",
            "--no-owner",
            "--no-acl",
        ]
    )
    if replace:
        command.extend(["--clean", "--if-exists"])
    return command, env


def _run_archive(command: list[str], snapshot: Path, env: dict[str, str]):
    with snapshot.open("rb") as archive:
        # Capture stderr because PostgreSQL diagnostics may include connection data.
        result = subprocess.run(
            command,
            stdin=archive,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            "PostgreSQL archive validation or restoration failed; check the configured service and archive version."
        )


def _connect_admin(config: RestoreConfig):
    return psycopg2.connect(
        host=config.host,
        port=config.port,
        dbname=config.database,
        user=config.admin_user,
        password=config.admin_password,
        connect_timeout=10,
    )


def _check_target(config: RestoreConfig, *, replace: bool):
    with closing(_connect_admin(config)) as connection, connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON c.relnamespace = n.oid WHERE n.nspname !~ '^pg_' AND n.nspname <> 'information_schema' AND c.relkind IN ('r', 'p', 'v', 'm', 'S', 'f'))"
        )
        if cursor.fetchone()[0] and not replace:
            raise ValueError(
                "The target database is nonempty. Use --replace explicitly to replace snapshot objects."
            )
        cursor.execute(
            "SELECT r.rolsuper OR r.rolcreatedb OR r.rolcreaterole OR r.rolreplication OR r.rolbypassrls, EXISTS (SELECT 1 FROM pg_auth_members m WHERE m.member = r.oid), EXISTS (SELECT 1 FROM pg_class c WHERE c.relowner = r.oid) OR EXISTS (SELECT 1 FROM pg_namespace n WHERE n.nspowner = r.oid) OR EXISTS (SELECT 1 FROM pg_database d WHERE d.datdba = r.oid) FROM pg_roles r WHERE r.rolname = %s",
            (config.app_user,),
        )
        existing = cursor.fetchone()
        if existing and any(existing):
            raise ValueError(
                "The existing application role has elevated permissions, memberships, or owned objects. Configure a separate unprivileged reader role."
            )
        # PUBLIC has grantee OID 0 and is absent from pg_roles. Column ACLs
        # survive table-level REVOKE, so both need independent preflight checks.
        cursor.execute(
            "WITH objects AS ("
            "SELECT n.nspname, c.relname AS object_name, 'table' AS kind, c.relacl AS acl "
            "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "UNION ALL SELECT n.nspname, c.relname, 'column', a.attacl "
            "FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE a.attnum > 0 AND NOT a.attisdropped "
            "UNION ALL SELECT n.nspname, p.proname, 'function', p.proacl "
            "FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
            "UNION ALL SELECT n.nspname, n.nspname, 'schema', n.nspacl FROM pg_namespace n"
            ") SELECT EXISTS ("
            "SELECT 1 FROM objects CROSS JOIN LATERAL aclexplode(objects.acl) grants "
            "WHERE objects.nspname !~ '^pg_' AND objects.nspname <> 'information_schema' AND ("
            "(grants.grantee = 0 AND ("
            "(objects.kind IN ('table', 'column') AND (objects.nspname <> 'public' "
            "OR objects.object_name <> ALL(%s) OR grants.privilege_type <> 'SELECT')) "
            "OR (objects.kind NOT IN ('table', 'column') AND objects.nspname <> 'public'))) "
            "OR (grants.grantee = (SELECT oid FROM pg_roles WHERE rolname = %s) AND ("
            "objects.nspname <> 'public' OR (objects.kind = 'column' AND ("
            "objects.object_name <> ALL(%s) OR grants.privilege_type <> 'SELECT'))))"
            ")) OR EXISTS ("
            "SELECT 1 FROM pg_default_acl defaults CROSS JOIN LATERAL aclexplode(defaults.defaclacl) grants "
            "WHERE grants.grantee = 0 OR grants.grantee = (SELECT oid FROM pg_roles WHERE rolname = %s))",
            (list(WAREHOUSE_OBJECTS), config.app_user, list(WAREHOUSE_OBJECTS), config.app_user),
        )
        if cursor.fetchone()[0]:
            raise ValueError(
                "The application role or PUBLIC has incompatible privileges outside the warehouse or future default grants. Configure an isolated database and dedicated reader role."
            )


def provision_reader(config: RestoreConfig):
    """Grant an independent login SELECT on only the nine warehouse objects."""
    role = sql.Identifier(config.app_user)
    database = sql.Identifier(config.database)
    with closing(_connect_admin(config)) as connection, connection, connection.cursor() as cursor:
        cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (config.app_user,))
        if cursor.fetchone() is None:
            cursor.execute(sql.SQL("CREATE ROLE {} LOGIN").format(role))
        cursor.execute(
            sql.SQL(
                "ALTER ROLE {} WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS NOINHERIT PASSWORD {}"
            ).format(role, sql.Literal(config.app_password))
        )
        cursor.execute(
            sql.SQL("REVOKE ALL PRIVILEGES ON DATABASE {} FROM {}").format(database, role)
        )
        cursor.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(database, role))
        # Public defaults would otherwise grant CREATE/TEMP through every role.
        cursor.execute(
            sql.SQL("REVOKE CREATE, TEMPORARY ON DATABASE {} FROM PUBLIC").format(database)
        )
        cursor.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        cursor.execute(sql.SQL("REVOKE ALL PRIVILEGES ON SCHEMA public FROM {}").format(role))
        cursor.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(role))
        cursor.execute(
            sql.SQL("REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM {}").format(role)
        )
        cursor.execute(
            sql.SQL("REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM {}").format(role)
        )
        objects = sql.SQL(", ").join(sql.Identifier("public", name) for name in WAREHOUSE_OBJECTS)
        cursor.execute(sql.SQL("GRANT SELECT ON {} TO {}").format(objects, role))
        cursor.execute(
            sql.SQL("ALTER ROLE {} IN DATABASE {} SET default_transaction_read_only = on").format(
                role, database
            )
        )
        cursor.execute(
            sql.SQL("ALTER ROLE {} IN DATABASE {} SET statement_timeout = '10s'").format(
                role, database
            )
        )
        cursor.execute(
            sql.SQL("ALTER ROLE {} IN DATABASE {} SET search_path = pg_catalog").format(
                role, database
            )
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot", type=Path, default=PROJECT_ROOT / "data/fraudData/fraudData_snapshot.dump"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Explicitly clean snapshot objects in an existing nonempty database.",
    )
    args = parser.parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    try:
        snapshot = validate_snapshot(args.snapshot)
        config = RestoreConfig.from_environment()
        config.validate()
        executable = shutil.which("pg_restore")
        if executable is None and shutil.which("docker") is None:
            raise ValueError(
                "Install pg_restore or provide Docker access to the configured running PostgreSQL container."
            )
        command, env = restore_command(config, executable=executable, replace=args.replace)
        list_command = (
            [executable, "--list"]
            if executable
            else ["docker", "exec", "-i", config.container, "pg_restore", "--list"]
        )
        _run_archive(list_command, snapshot, env)
        _check_target(config, replace=args.replace)
        _run_archive(command, snapshot, env)
        provision_reader(config)
    except (OSError, ValueError, RuntimeError, psycopg2.Error):
        # Never print driver errors, command environments, or password-bearing SQL.
        raise SystemExit(
            "Database initialization failed. Validate the PGDMP archive, separate administrator/reader credentials, service readiness, and --replace choice."
        ) from None
    print("Snapshot restored atomically; warehouse reader role provisioned.")


if __name__ == "__main__":
    main()
