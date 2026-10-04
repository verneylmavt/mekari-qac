# Challenge improvement plan

Goal: deliver a polished fraud Q&A challenge submission, with accurate evidence from the synthetic transaction warehouse and both original fraud PDFs.
Context: `assets/Mekari - AI Engineer.pdf`, the existing FastAPI/LangGraph and Streamlit application, PostgreSQL snapshot, Bhatla PDF, and EBA 2024 report.
Constraints: implement directly on main; preserve entrypoints and original artifacts; no paid API calls during validation; versioned embeddings in Git LFS; synchronous bounded execution; commit and push the finished work.
Done when: offline tests, lint, corpus integrity/evaluation and browser checks pass; documentation matches behavior; logical commits are pushed to origin/main.

## Commit sequence and verification

1. [ ] Reproducible Python 3.11 configuration, dependency lock, injectable resources, safe imports, environment hygiene. Verify import/lifespan and schema tests.
2. [ ] Read-only SQL AST policy, warehouse role, bounded transactions and results, safe restoration. Verify hostile SQL and database boundary tests.
3. [ ] Canonical page-aware corpus for both PDFs, stable IDs, manifest and vectors, safe Qdrant alias initialization. Verify source facts and artifact integrity.
4. [ ] Hybrid retrieval, reranking, complete token-bounded evidence, document filters and versioned caches. Verify ranking/filter/cache behavior.
5. [ ] Structured conversational planner, mixed answers, citation validation, clarification/abstention and transparent evidence quality. Verify deterministic stub scenarios and failure behavior.
6. [ ] Admission/inference limits, deadlines, timeouts/retry, readiness and safe failures. Verify concurrent saturation and resource cleanup.
7. [ ] Responsive accessible Streamlit experience, source controls, error recovery, export and SQL-backed charts. Verify client tests, AppTest and browser fixture.
8. [ ] Offline evaluation, CI, demo screenshots and updated README/paired contributor guides. Run fresh independent review and all checks, then push.

## Shared interfaces

`ChatRequest`: question (1..4000 characters), nullable history (max 12 user/assistant messages, each max 8000 characters), nullable document_id (`bhatla`/`eba`). Auto is null.
`ChatResponse`: existing answer/answer_type/quality_score/sql/sources, plus status (`answered`/`clarification`/`insufficient_evidence`), quality_available, quality_method, quality_breakdown, request_id, elapsed_ms, truncated, timings. answer_type supports data/document/mixed/other.
Errors: `detail` object with code, message, retryable, request_id. Saturation/dependency 503, deadline 504, validation 422.
SQL boundary: `run_sql_query(sql, *, engine=None, max_rows=200, timeout_ms=10000)` returns an object with sql, rows, truncated; `SQLPolicyError` and `SQLExecutionError` expose safe messages/codes (repairable only known syntax/schema errors).
Retrieval boundary: `DocumentRetriever(settings, inference_gate=None)`; `retrieve(question, document_id=None, deadline=None)` returns chunks with citation_id, payload {chunk_id, document_id, title, page, section, text, source_sha256}, rerank_score. `ready()` validates alias/profile/artifacts/local model availability; close() releases resources. Runtime imports must not load/download weights. Query prefix is the full BGE English instruction; passages are plain text.
All chat work remains synchronous. Defaults: chat admission 4, inference 1, request 120s, external call 30s, SQL 10s, rows 200, prompt SQL rows 20, retrieval candidates 30/method, final chunks 8, document context 6000 tokens.

## Decisions and progress

Ruling: user explicitly authorized main and push; use disjoint worker file ownership in this checkout instead of worktrees.
Ruling: preserve legacy notebooks/exports as historical preparation; new safe scripts replace their runtime setup responsibilities.
Ruling: live Docker services are unavailable at preflight; use isolated fakes and local Qdrant tests, clearly report integration limits.
Ruling: use pinned packages and a separate ignored .venv; dependency and public model downloads are setup, while evaluation performs zero paid API calls.

| Shared work | Producer / consumer | Resolution |
| --- | --- | --- |
| Foundation / all components | Settings and response contracts | Root owns config/schemas/runtime; workers follow interfaces above. |
| SQL / orchestration | Bounded query result and safe errors | Root uses result attributes; policy errors never repaired. |
| Corpus / retrieval | Manifest, chunks and profile | One worker owns both with separate implementation commits. |
| Orchestration / frontend | Extended response fields | Frontend handles absent optional metadata for compatibility. |
| Every task / verification | Implemented behavior versus tests | Each component supplies focused offline tests before completion. |

The original requirement PDF and approved conversation plan are the specification; this file records implementation and verification status.
