"""Initializer tests never touch Docker or a live database."""

import pytest

from scripts import init_postgresql as initializer


def test_lfs_pointer_or_invalid_dump_is_rejected_before_external_work(tmp_path):
    snapshot = tmp_path / "snapshot.dump"
    snapshot.write_bytes(b"version https://git-lfs.github.com/spec/v1")
    with pytest.raises(ValueError, match="PGDMP"):
        initializer.validate_snapshot(snapshot)


def test_valid_dump_header_is_accepted(tmp_path):
    snapshot = tmp_path / "snapshot.dump"
    snapshot.write_bytes(b"PGDMP" + b"\x00" * 30)
    assert initializer.validate_snapshot(snapshot) == snapshot.resolve()


def test_restore_is_atomic_and_clean_requires_explicit_replace():
    config = initializer.RestoreConfig(
        host="db.example",
        port=6543,
        database="warehouse",
        admin_user="admin",
        admin_password="secret",
        app_user="reader",
        app_password="app-secret",
        container="configured-db",
    )
    command, env = initializer.restore_command(config, executable="pg_restore", replace=False)
    assert "--single-transaction" in command and "--exit-on-error" in command
    assert "--no-owner" in command and "--no-acl" in command
    assert "--clean" not in command
    assert "db.example" in command and "6543" in command and "warehouse" in command
    assert "secret" not in " ".join(command)
    assert env["PGPASSWORD"] == "secret"
    command, _ = initializer.restore_command(config, executable=None, replace=True)
    assert command[:3] == ["docker", "exec", "-i"]
    assert "configured-db" in command and "--clean" in command and "--if-exists" in command
    assert "db.example" in command and "6543" in command
    assert "secret" not in " ".join(command)


def test_reader_may_not_be_admin():
    config = initializer.RestoreConfig(
        host="localhost",
        port=5432,
        database="db",
        admin_user="same",
        admin_password="secret",
        app_user="same",
        app_password="secret",
        container="postgresql",
    )
    with pytest.raises(ValueError, match="separate"):
        config.validate()


def test_builtin_postgres_reader_role_is_rejected_before_restore():
    config = initializer.RestoreConfig(
        "localhost", 5432, "db", "admin", "secret", "pg_read_all_data", "secret", "postgresql"
    )
    with pytest.raises(ValueError, match="reserved"):
        config.validate()


class AdminCursor:
    def __init__(self, answers):
        self.answers = iter(answers)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, command, params=None):
        pass

    def fetchone(self):
        return next(self.answers)


class AdminConnection:
    def __init__(self, answers):
        self.admin_cursor = AdminCursor(answers)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return self.admin_cursor

    def close(self):
        self.closed = True


def test_nonempty_database_requires_replace_and_closes_preflight_connection(monkeypatch):
    connection = AdminConnection([(True,)])
    monkeypatch.setattr(initializer, "_connect_admin", lambda _: connection)
    with pytest.raises(ValueError, match="--replace"):
        initializer._check_target(None, replace=False)
    assert connection.closed


@pytest.mark.parametrize(
    "privileged, memberships, owner",
    [(True, False, False), (False, True, False), (False, False, True)],
)
def test_existing_privileged_reader_rejected_even_for_explicit_replacement(
    monkeypatch, privileged, memberships, owner
):
    connection = AdminConnection([(True,), (privileged, memberships, owner)])
    monkeypatch.setattr(initializer, "_connect_admin", lambda _: connection)
    config = initializer.RestoreConfig(
        "localhost", 5432, "db", "admin", "secret", "reader", "secret", "postgresql"
    )
    with pytest.raises(ValueError, match="unprivileged"):
        initializer._check_target(config, replace=True)
    assert connection.closed


def test_existing_reader_with_direct_privileges_elsewhere_is_rejected(monkeypatch):
    connection = AdminConnection([(True,), (False, False, False), (True,)])
    monkeypatch.setattr(initializer, "_connect_admin", lambda _: connection)
    config = initializer.RestoreConfig(
        "localhost", 5432, "db", "admin", "secret", "reader", "secret", "postgresql"
    )
    with pytest.raises(ValueError, match="outside"):
        initializer._check_target(config, replace=True)
    assert connection.closed


def test_invalid_dump_never_opens_database_or_runs_external_process(tmp_path, monkeypatch):
    snapshot = tmp_path / "pointer.dump"
    snapshot.write_text("version https://git-lfs.github.com/spec/v1")

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid dump reached an external boundary")

    monkeypatch.setattr(initializer, "_connect_admin", forbidden)
    monkeypatch.setattr(initializer.subprocess, "run", forbidden)
    with pytest.raises(SystemExit):
        initializer.main(["--snapshot", str(snapshot), "--replace"])


def test_preflight_failure_never_restores_or_provisions_reader(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot.dump"
    snapshot.write_bytes(b"PGDMP" + b"\x00" * 30)
    events = []
    config = initializer.RestoreConfig(
        "localhost", 5432, "db", "admin", "secret", "reader", "app-secret", "postgresql"
    )
    monkeypatch.setattr(initializer.RestoreConfig, "from_environment", lambda: config)
    monkeypatch.setattr(initializer.shutil, "which", lambda _: "pg_restore")
    monkeypatch.setattr(initializer, "_run_archive", lambda command, *_: events.append(command))

    def nonempty(*args, **kwargs):
        raise ValueError("nonempty")

    def forbidden(*args, **kwargs):
        pytest.fail("Failed preflight still provisioned a reader")

    monkeypatch.setattr(initializer, "_check_target", nonempty)
    monkeypatch.setattr(initializer, "provision_reader", forbidden)
    with pytest.raises(SystemExit):
        initializer.main(["--snapshot", str(snapshot)])
    assert events == [["pg_restore", "--list"]]
