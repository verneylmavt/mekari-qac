# Canonical document corpus

This directory contains prebuilt embeddings of the original `Bhatla.pdf` (17 PDF pages)
and the EBA/ECB 2024 payment-fraud report (35 PDF pages). The historical DOCX summaries,
notebooks, JSON chunks and NPY vectors remain separate and are never mixed into this corpus.

`chunks.json` is readable evidence with stable UUID5 IDs, source SHA-256, document title,
physical PDF page, section, chunk position, token count and matching vector row.
`vectors.npy` is a Git LFS artifact containing actual normalized float32 BGE embeddings.
Run `git lfs pull` before setup; a pointer file fails integrity validation.
`manifest.json` records exact source, weight, embedding-profile, artifact and row-order hashes.

The embedding profile uses `BAAI/bge-base-en-v1.5` with plain passages and the query instruction
`Represent this sentence for searching relevant passages: `. Its exact model revision and
the `BAAI/bge-reranker-base` revision are pinned. Chunks stay on one PDF page and within
420 tokens under both tokenizers; final evidence retains complete chunks.

PDF page numbers are one-based physical pages, not printed page labels. Tables retain
sorted extraction spacing. EBA chart labels carry a caveat because plain text cannot reliably
pair chart series, periods and numeric labels. The corpus never assigns those pairs by inference.
Bhatla PDF page 15 includes an explicit visual transcription of Figure 2 and preserves its
0.08% excessive-review fraud-loss label alongside the conflicting 0.06% prose value.

From the repository root, rebuild only during an intentional setup operation:

```powershell
.venv/Scripts/python.exe scripts/build_corpus.py --cache-models-only
.venv/Scripts/python.exe scripts/build_corpus.py
.venv/Scripts/python.exe scripts/build_corpus.py --offline
.venv/Scripts/python.exe scripts/init_qdrant.py
```

The first build may download the pinned public model snapshots. Runtime only accepts complete
cached snapshots and never downloads weights. The initializer validates every local artifact
before a remote operation, uploads to a versioned collection, verifies payloads/vectors, and
atomically switches the configured `fraud_documents` alias. Existing collections are retained;
the script does not start Docker or delete an active collection.
