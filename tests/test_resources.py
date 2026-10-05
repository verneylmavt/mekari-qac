from contextlib import nullcontext

from backend.app.config import Settings
from backend.app.resources import Resources


def test_readiness_rejects_role_without_select_or_schema_usage():
    class Result:
        def __init__(self, value):
            self.value = value

        def scalar(self):
            return self.value

        def one(self):
            return (False, False, False, False)

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def begin(self):
            return nullcontext()

        def execute(self, sql, parameters=None):
            query = str(sql)
            if "to_regclass" in query:
                return Result("public.dim_date")
            return Result(False)

    class Engine:
        def connect(self):
            return Connection()

    class Retriever:
        def ready(self):
            return {"qdrant_ok": True, "models_cached": True}

    resources = Resources(Settings(_env_file=None, openai_api_key="fixture-not-used"))
    resources._engine = Engine()
    resources._retriever = Retriever()
    assert not resources.ready()["db_ok"]
