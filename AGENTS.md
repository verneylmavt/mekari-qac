# Contributor guide

## Scope and architecture

This is the Mekari fraud Q&A challenge: Streamlit → FastAPI → a compiled LangGraph planner → PostgreSQL, original-PDF retrieval, or both → grounded answer → evidence quality rubric. Accuracy and source transparency take priority. See README.md for setup/contracts and PLANS.md for the completed improvement plan.

Runtime code lives in backend/app and frontend. Safe setup/evaluation entrypoints live in scripts. Tests use injected dependencies and run without paid calls. Keep AGENTS.md and CLAUDE.md byte-identical whenever either changes.

## Essential invariants

- Imports and /live must remain free of connections, model loading, and model downloads. Resources are lazy, lifespan-owned, synchronized, and closed on shutdown.
- Keep synchronous work bounded: chat admission, inference slots, monotonic deadlines, provider timeout/retry, DB pool/statement timeout, result size, retrieval context, and history. Limits are per process; use one backend worker. Native inference holds its slot until it actually finishes.
- Never log questions, document excerpts, keys, passwords, driver error text, or provider response bodies. Return safe errors with request IDs. Settings secrets use SecretStr; construct DB URLs with URL.create.
- SQL policy is a parsed PostgreSQL allowlist, not a text search. One read-only SELECT/set operation, scoped CTEs, nine public warehouse objects, explicitly allowed functions/casts. Policy failures are never repaired. Known syntax/schema failures permit one repair, validated again.
- Execute raw validated SQL with exec_driver_sql(no_parameters=True), public-qualified tables, search_path=pg_catalog, read-only transactions, bounded fetch, and unconditional rollback. Preserve colon/percent string literals. Keep effective PUBLIC/column/default privilege checks in reader provisioning.
- Fraud rates/value shares are fractions. Aggregate by summing numerator and denominator, never averaging rates. Distinguish counts, count rates, fraud value, and value shares; trends sort chronologically. Amount currency is unspecified: label dataset units.
- Document provenance comes from the original Bhatla and EBA/ECB PDFs. Cite physical one-based PDF pages and stable chunk IDs. Do not merge legacy DOCX summaries or old embeddings into the canonical corpus.
- BGE uses plain passages and the full query instruction in the manifest. Model revisions, weight hashes, dimensions, normalization, artifact hashes, ID/row order, source hashes, and active alias must agree. Runtime uses cached pinned snapshots only.
- Canonical JSON must use LF on every platform. The builder writes explicit newline="\n" and .gitattributes enforces it; raw file hashes must match Git checkout bytes, not just the current working copy.
- Retrieval unions dense and BM25 candidates then reranks. Keep complete chunks within both tokenizer and context limits. Cache by resolved question, document filter, corpus/profile/version/alias; do not cache answers or generated SQL. Serialize cold model initialization.
- Every substantive answer paragraph must cite supplied evidence. Numeric validation uses that paragraph's cited sources. Mixed answers require both evidence branches. Preserve source population/period/unit distinctions and abstain or clarify when unsupported.
- Bhatla page15 Figure2 has 0.08% versus 0.06% in prose: report the conflict. EBA H1 2023 cross-border card fraud is 71% by value, 68% by volume. Unpaired chart labels do not establish a series/period mapping.
- Quality is a weighted evidence rubric, not a probability. The judge receives exactly the answer evidence. When unavailable, quality_available=false and the UI shows N/A; zero is only the compatibility sentinel.
- Streamlit requests use pending turn IDs; one submission per rerun. Disable controls while pending, retain original retry context, and exclude failed turns from backend history. Charts must derive only from returned SQL fields. Demo mode must be explicitly labelled.

## Setup and artifacts

Use Windows/Python3.11 from repository root. requirements.in is the direct dependency source; requirements.txt is the compiled lock for this platform. Backend/frontend manifests include it. Preserve local .env; it is ignored. Use .env.example and separate administrator/warehouse-reader credentials. Never commit secrets.

Run git lfs pull before initialization. Canonical data/corpus/vectors.npy and the warehouse dump require real LFS content, not pointer files. Preserve raw data, PDFs, snapshots, demo media, and historical outputs unless regeneration is explicitly part of the task.

scripts/init_postgresql.py validates the archive before mutation and requires an empty target unless --replace is explicit. scripts/init_qdrant.py verifies local artifacts, uploads/verifies a versioned collection, then atomically switches the alias; it retains previous collections. Neither script starts Docker. Do not apply setup operations to a user's existing service just to verify code.

Historical data notebooks and Python exports perform destructive operations at module scope. Do not import them. Their separate manifests describe preparation only; new runtime/setup code must not depend on them. Materialized warehouse views are snapshots and do not refresh automatically.

## Verification

```powershell
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m ruff check backend frontend scripts tests
.venv/Scripts/python.exe -m ruff format --check backend frontend scripts tests
.venv/Scripts/python.exe -m pytest -q
```

Default tests are offline; CI fetches only the canonical vector LFS artifact. Opt-in real SQL tests require FRAUD_TEST_LOCAL_POSTGRES=1 and a disposable local PostgreSQL at127.0.0.1:15432 with postgres administrator. They create/remove unique databases/roles and never use application DB settings. Local integration was checked on PostgreSQL18; Compose targets16.

scripts/evaluate_retrieval.py uses real cached models and in-memory Qdrant for seven documentary probes; assets/evaluation/retrieval_report.json reports measured retrieval only. No offline result establishes live planner/SQL/answer accuracy. Do not spend provider calls without a budgeted request.

For frontend changes run the client/helper/AppTest suite and verify desktop/mobile states. scripts.demo_backend:app plus FRAUD_DEMO_MODE=1 supports clearly labelled offline browser checks. Keep temporary logs, downloads, screenshots not intended as artifacts, model caches, and test databases out of commits. .gitignore covers .venv, test-results, tool outputs, and logs.

Before completion inspect the whole diff for regressions, check matching guide hashes and git diff --check, and report validation limits accurately. Commit/push only when requested; avoid destructive Git operations or unrelated data regeneration.
