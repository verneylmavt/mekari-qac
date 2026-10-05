"""Explicit public-model setup: rebuild canonical PDF chunks and actual BGE vectors.

Run from the repository root. Runtime never invokes this script or downloads weights.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.vdb.corpus import (  # noqa: E402 - support direct script execution
    QUERY_INSTRUCTION,
    canonical_hash,
    file_hash,
    load_corpus,
    stable_chunk_id,
)

SOURCES = {
    "bhatla": (
        "data/Understanding Credit Card Frauds/Bhatla.pdf",
        "Understanding Credit Card Frauds",
    ),
    "eba": (
        "data/2024 REPORT ON PAYMENT FRAUD/EBA_ECB 2024 Report on Payment Fraud.pdf",
        "2024 Report on Payment Fraud (EBA and ECB)",
    ),
}
EMBED_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
RERANK_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"
EMBED_WEIGHTS_SHA256 = "c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7"
RERANK_WEIGHTS_SHA256 = "ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd"
PUBLICATION_DATES = {"bhatla": "June 2003", "eba": "2024"}
FIGURE_2 = (
    "[Visual transcription, Figure 2: minimizing the total cost of fraud. "
    "Insufficient screening: 2.0% orders reviewed, 1.0% fraud losses; balanced screening: "
    "5.0% reviewed, 0.3% fraud losses; excessive reviews: 30.0% reviewed, 0.08% fraud losses. "
    "The figure caption describes percentages of orders reviewed and fraudulent orders. "
    "Source inconsistency: the prose on this same page states 0.06% for the excessive-review "
    "fraud loss, while Figure 2 labels it 0.08%. Preserve both; do not silently reconcile them.]"
)


def clean_lines(text):
    lines = []
    for raw in text.replace("\ufffd", "-").splitlines():
        # MuPDF's sorted text spaces retain table columns and chart boundaries.
        line = raw.strip()
        if (
            not line
            or re.fullmatch(r"Page \d+ of \d+", line)
            or line == "Understanding Credit Card Frauds"
        ):
            continue
        lines.append(line)
    return lines


def extract_chunks(root, tokenizer, max_tokens=420, rerank_tokenizer=None):
    import pymupdf

    def count(text):
        sizes = [len(tokenizer.encode(text, add_special_tokens=True, verbose=False))]
        if rerank_tokenizer is not None:
            sizes.append(len(rerank_tokenizer.encode(text, add_special_tokens=True, verbose=False)))
        return max(sizes)

    chunks, sources = [], {}
    for document_id, (relative, title) in SOURCES.items():
        path = root / relative
        source_hash = file_hash(path)
        with pymupdf.open(path) as pdf:
            sources[document_id] = {
                "path": relative,
                "title": title,
                "sha256": source_hash,
                "pages": len(pdf),
                "publication_date": PUBLICATION_DATES[document_id],
            }
            section = title
            for page_no, page in enumerate(pdf, 1):
                lines = clean_lines(page.get_text(sort=True))
                headings = []
                for block in page.get_text("dict")["blocks"]:
                    for line in block.get("lines", []):
                        spans = line["spans"]
                        text = " ".join(s["text"].strip() for s in spans).strip()
                        heading = any(
                            (s["size"] >= 10.5 and s["flags"] & 16)
                            if document_id == "bhatla"
                            else s["size"] >= 18
                            for s in spans
                        )
                        if text and len(text) < 120 and heading:
                            if text not in (title, "Understanding Credit Card Frauds"):
                                headings.append(text)
                text = "\n".join(lines)
                if document_id == "bhatla" and page_no == 15:
                    text += "\n" + FIGURE_2
                caveat = ""
                if document_id == "eba" and re.search(r"Chart \d+", text):
                    caveat = (
                        "[Chart extraction caveat: numeric labels are unpaired; do not infer series, "
                        "periods or value/volume mappings from their order. Use explicit narrative statements.]\n"
                    )
                units = []
                for line in text.splitlines():
                    if count(line) <= max_tokens - 48:
                        units.append(line)
                    else:
                        segment = []
                        for word in line.split():
                            if count(" ".join(segment + [word])) > max_tokens - 48:
                                units.append(" ".join(segment))
                                segment = []
                            segment.append(word)
                        if segment:
                            units.append(" ".join(segment))
                buffer, index = [], 0
                for unit in units + [None]:
                    prefix = (
                        f"{title} | Published {PUBLICATION_DATES[document_id]} | "
                        f"PDF page {page_no} | {section}\n"
                    ) + caveat
                    if buffer and (
                        unit is None
                        or unit in headings
                        or count(prefix + "\n".join(buffer + [unit])) > max_tokens
                    ):
                        chunk_text = prefix + "\n".join(buffer)
                        chunk = {
                            "document_id": document_id,
                            "publication_date": PUBLICATION_DATES[document_id],
                            "title": title,
                            "page": page_no,
                            "section": section,
                            "chunk_index": index,
                            "text": chunk_text,
                            "source_sha256": source_hash,
                            "token_count": count(chunk_text),
                            "vector_row": len(chunks),
                        }
                        chunk["chunk_id"] = stable_chunk_id(chunk)
                        chunks.append(chunk)
                        index += 1
                        buffer = []
                    if unit is not None:
                        if unit in headings:
                            section = unit
                        buffer.append(unit)
    return chunks, sources


def build(output, *, offline=False, cache_models_only=False):
    import torch

    torch.set_num_threads(4)
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    embed_path = snapshot_download(
        "BAAI/bge-base-en-v1.5",
        revision=EMBED_REVISION,
        local_files_only=offline,
        ignore_patterns=["*.onnx", "*.xml", "*.bin", "onnx/*", "openvino/*"],
    )
    rerank_path = snapshot_download(
        "BAAI/bge-reranker-base",
        revision=RERANK_REVISION,
        local_files_only=offline,
        ignore_patterns=["*.onnx", "*.xml", "*.bin", "onnx/*", "openvino/*"],
    )
    embed_weights_hash = file_hash(Path(embed_path) / "model.safetensors")
    rerank_weights_hash = file_hash(Path(rerank_path) / "model.safetensors")
    if embed_weights_hash != EMBED_WEIGHTS_SHA256 or rerank_weights_hash != RERANK_WEIGHTS_SHA256:
        raise ValueError("Cached weights do not match the pinned public Hugging Face model hashes")
    if cache_models_only:
        print("Pinned embedding and reranker snapshots cached and weight hashes verified")
        return
    model = SentenceTransformer(embed_path, device="cpu", local_files_only=True)
    profile = {
        "name": "bge-english-v1.5-plain-passage-v2",
        "model": "BAAI/bge-base-en-v1.5",
        "model_revision": Path(embed_path).name,
        "reranker": "BAAI/bge-reranker-base",
        "reranker_revision": Path(rerank_path).name,
        "dimension": 768,
        "normalized": True,
        "model_weights_sha256": embed_weights_hash,
        "reranker_weights_sha256": rerank_weights_hash,
        "passage_instruction": "",
        "query_instruction": QUERY_INSTRUCTION,
    }
    rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_path, local_files_only=True)
    chunks, sources = extract_chunks(ROOT, model.tokenizer, rerank_tokenizer=rerank_tokenizer)
    profile_hash = canonical_hash(profile)
    chunking = {
        "method": "page-lines-heading-context-v1",
        "max_tokens": 420,
        "tokenizer": "max-of-bge-and-reranker",
        "tokenizer_revision": profile["model_revision"],
        "reranker_tokenizer_revision": profile["reranker_revision"],
        "overlap": 0,
    }
    identity = {
        "profile_sha256": profile_hash,
        "point_order_sha256": canonical_hash([c["chunk_id"] for c in chunks]),
        "sources": sources,
        "chunking": chunking,
    }
    vectors = model.encode(
        [c["text"] for c in chunks],
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "vectors.npy", vectors, allow_pickle=False)
    identity["vectors_sha256"] = file_hash(output / "vectors.npy")
    version = canonical_hash(identity)[:16]
    for chunk in chunks:
        chunk.update(profile_sha256=profile_hash, artifact_version=version)
    (output / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "artifact_version": version,
        **identity,
        "embedding_profile": profile,
        "chunk_count": len(chunks),
        "artifacts": {
            name: {"sha256": file_hash(output / name)} for name in ("chunks.json", "vectors.npy")
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    load_corpus(output, verify_sources=True)
    print(
        f"Validated {len(chunks)} chunks, {vectors.shape[1]} normalized dimensions, artifact {version}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline", action="store_true", help="Require cached public model snapshots"
    )
    parser.add_argument(
        "--cache-models-only",
        action="store_true",
        help="Cache/validate weights without rebuilding prebuilt artifacts",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/corpus")
    args = parser.parse_args()
    build(args.output, offline=args.offline, cache_models_only=args.cache_models_only)
