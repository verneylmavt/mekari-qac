#!/usr/bin/env python
# coding: utf-8

# ## Indexing

# ### Import

# In[153]:


import os
import re
import uuid
from dataclasses import dataclass
from typing import List, Dict, Optional

import json
import numpy as np
from tqdm.auto import tqdm

from docx import Document

import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

from qdrant_client import QdrantClient
from qdrant_client.http.models import QueryResponse
from qdrant_client.http import models as qmodels


# ### Config & Variables

# In[109]:


docx_path = "Bhatla_Description.docx"

qdrant_url = "http://localhost:6333"
qdrant_collection = "bhatla_credit_fraud"

embed_model_name = "BAAI/bge-base-en-v1.5"
embed_dim = 768

reranker_model_name = "BAAI/bge-reranker-base"

max_tokens_per_chunk = 256
chunk_overlap_sentences = 1


# In[110]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ### Dataclass

# In[111]:


# for Paragraph or Heading
@dataclass
class RawElement:
    type: str
    level: Optional[int]
    text: str

# for Section
@dataclass
class Block:
    section: Optional[str]
    subsection: Optional[str]
    text: str
    block_index: int

# for Text
@dataclass
class Chunk:
    chunk_id: str
    section: Optional[str]
    subsection: Optional[str]
    block_index: int
    chunk_index: int
    text: str


# In[112]:


bullet_chars = ["•", "·", "●", "■", "▪", "¤", "-", "–", "—"]

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_all_caps(text: str) -> bool:
    stripped = re.sub(r"[^A-Za-z]", "", text)
    return stripped.isupper() and len(stripped) > 3


# ### Data Pre-Processing

# #### Data Parsing

# In[113]:


def parse_docx_to_raw_elements(docx_path: str) -> List[RawElement]:
    doc = Document(docx_path)
    elements: List[RawElement] = []

    for para in doc.paragraphs:
        text = clean_text(para.text)
        if not text:
            continue

        style_name = para.style.name if para.style else ""

        level = None
        elem_type = "paragraph"

        if style_name.startswith("Heading"):
            elem_type = "heading"
            try:
                level = int(style_name.split()[-1])
            except ValueError:
                level = 1
        else:
            if is_all_caps(text) and len(text.split()) <= 6:
                elem_type = "heading"
                level = 2

        elements.append(RawElement(type=elem_type, level=level, text=text))

    return elements


# In[114]:


raw_elements = parse_docx_to_raw_elements(docx_path)
print(f"Number of Raw Elements: {len(raw_elements)}")
print("First 5 Raw Elements:")
for el in raw_elements[:5]:
    print(f"  type={el.type}, level={el.level}, text='{el.text[:80]}...'")


# In[115]:


def build_blocks_from_elements(elements: List[RawElement]) -> List[Block]:
    blocks: List[Block] = []
    current_section: Optional[str] = None
    current_subsection: Optional[str] = None
    current_paragraphs: List[str] = []
    block_index = 0

    def flush_block():
        nonlocal block_index, current_paragraphs
        if current_paragraphs:
            text = " ".join(current_paragraphs).strip()
            blocks.append(
                Block(
                    section=current_section,
                    subsection=current_subsection,
                    text=text,
                    block_index=block_index,
                )
            )
            block_index += 1
            current_paragraphs = []

    for el in elements:
        if el.type == "heading":
            flush_block()
            if el.level is None or el.level == 1:
                current_section = el.text
                current_subsection = None
            elif el.level == 2:
                current_subsection = el.text
            else:
                current_paragraphs.append(el.text)
        else:
            current_paragraphs.append(el.text)

    flush_block()

    return blocks


# In[116]:


blocks = build_blocks_from_elements(raw_elements)
print(f"Number of Blocks: {len(blocks)}")
print("First 5 Blocks:")
for b in blocks[:5]:
    print(f"  block_index={b.block_index}, section={b.section}, subsection={b.subsection}")
    print(f"    text='{b.text[:120]}...'")


# In[140]:


total_block_chars = sum(len(b.text) for b in blocks)
total_block_words = sum(len(b.text.split()) for b in blocks)

print("Blocks")
print(f"  Number of Blocks: {len(blocks)}")
print(f"  Number of Characters: {total_block_chars}")
print(f"  Number of Words: {total_block_words}")
print(f"  Chars per Block: {total_block_chars / len(blocks):.2f}")
print(f"  Words per Block: {total_block_words / len(blocks):.2f}")


# #### Data Splitting

# In[117]:


def split_into_sentences(text: str) -> List[str]:
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


# In[118]:


def chunk_block(
    blk: Block,
    max_tokens: int = max_tokens_per_chunk,
    overlap_sentences: int = chunk_overlap_sentences,
) -> List[Chunk]:
    sentences = split_into_sentences(blk.text)
    chunks_list: List[Chunk] = []
    current_sentences: List[str] = []
    current_count = 0
    chunk_index = 0

    i = 0
    while i < len(sentences):
        s = sentences[i]
        num_tokens = len(s.split())

        if current_sentences and current_count + num_tokens > max_tokens:
            chunk_text = " ".join(current_sentences).strip()
            chunk_id = str(uuid.uuid4())
            chunks_list.append(
                Chunk(
                    chunk_id=chunk_id,
                    section=blk.section,
                    subsection=blk.subsection,
                    block_index=blk.block_index,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )
            chunk_index += 1

            overlap = current_sentences[-overlap_sentences:] if overlap_sentences > 0 else []
            current_sentences = overlap.copy()
            current_count = sum(len(sen.split()) for sen in current_sentences)

        current_sentences.append(s)
        current_count += num_tokens
        i += 1

    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        chunk_id = str(uuid.uuid4())
        chunks_list.append(
            Chunk(
                chunk_id=chunk_id,
                section=blk.section,
                subsection=blk.subsection,
                block_index=blk.block_index,
                chunk_index=chunk_index,
                text=chunk_text,
            )
        )

    return chunks_list


def build_all_chunks(blocks: List[Block]) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for blk in blocks:
        block_chunks = chunk_block(blk)
        all_chunks.extend(block_chunks)
    return all_chunks


# In[119]:


chunks = build_all_chunks(blocks)
print(f"Number of Chunks: {len(chunks)}")
print("Sample Chunk:")
print(f"  id={chunks[0].chunk_id}")
print(f"  section={chunks[0].section}, subsection={chunks[0].subsection}")
print(f"  text length (chars)={len(chunks[0].text)}")
print(f"  text snippet='{chunks[0].text[:200]}...'")


# In[142]:


chunk_word_lengths = [len(c.text.split()) for c in chunks]
chunk_char_lengths = [len(c.text) for c in chunks]

print("Chunks")
print(f"  Number of Chunks: {len(chunks)}")
print(f"  Number of Characters: {sum(chunk_char_lengths)}")
print(f"  Number of Words: {sum(chunk_word_lengths)}")
print(f"  Chars per Chunk: {np.mean(chunk_char_lengths):.2f}")
print(f"  Words per Chunk: {np.mean(chunk_word_lengths):.2f}")
print(f"  Min. Words per Chunk: {np.min(chunk_word_lengths)}")
print(f"  Max. Words per Chunk: {np.max(chunk_word_lengths)}")


# #### Exporting: Chunk

# In[154]:


chunk_records = [
    {
        "chunk_id": c.chunk_id,
        "section": c.section,
        "subsection": c.subsection,
        "block_index": c.block_index,
        "chunk_index": c.chunk_index,
        "text": c.text,
    }
    for c in chunks
]

with open("Bhatla_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunk_records, f, ensure_ascii=False, indent=2)


# ### Embedding: BAAI/bge-base-en-v1.5

# #### Embedding Model

# In[120]:


embed_model = SentenceTransformer(embed_model_name, device=device)
print(f"Embedding Model: '{embed_model_name}' on Device: {device}")


# #### Embedding Function

# In[121]:


def embed_texts(texts: List[str], batch_size: int = 32, is_query: bool = False) -> np.ndarray:
    if is_query:
        prefixed = [f"query: {t}" for t in texts]
    else:
        prefixed = [f"passage: {t}" for t in texts]
    embeddings = embed_model.encode(
        prefixed,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings

def embed_query(query: str) -> np.ndarray:
    return embed_texts([query], batch_size=1, is_query=True)[0]


# In[122]:


chunk_texts = [c.text for c in chunks]
print(f"Embedding {len(chunk_texts)} Chunks...")
chunk_embeddings = embed_texts(chunk_texts, batch_size=32, is_query=False)


# In[123]:


print("Embedding Shape:", chunk_embeddings.shape)
assert chunk_embeddings.shape[0] == len(chunks)
assert chunk_embeddings.shape[1] == embed_dim


# #### Exporting: Embedding

# In[143]:


np.save("Bhatla_embeddings.npy", chunk_embeddings)


# ### Reranker: BAAI/bge-reranker-base

# #### Reranker Model

# In[124]:


reranker = CrossEncoder(
    reranker_model_name,
    device=device,
    max_length=512,
    trust_remote_code=True,
)
print(f"Reranker Model: '{reranker_model_name}' on Device: {device}")


# #### Reranker Function

# In[125]:


def rerank_with_bge_reranker(
    query: str,
    retrieved_results: List[Dict],
    top_k: Optional[int] = None,
) -> List[Dict]:
    if not retrieved_results:
        return []

    texts = [r["payload"]["text"] for r in retrieved_results]
    pairs = [(query, t) for t in texts]

    scores = reranker.predict(pairs)

    for r, s in zip(retrieved_results, scores):
        r["rerank_score"] = float(s)

    reranked = sorted(retrieved_results, key=lambda x: x["rerank_score"], reverse=True)

    if top_k is not None:
        reranked = reranked[:top_k]

    return reranked


# ### Vector Store: Qdrant

# #### Vector Store Setup and Connection

# In[ ]:


import os
import subprocess

os.makedirs("qdrant", exist_ok=True)

if subprocess.run(["docker", "start", "qdrant"]).returncode != 0:
    subprocess.run([
        "docker", "run", "-d",
        "--name", "qdrant",
        "-p", "6333:6333",
        "-v", f"{os.getcwd()}\\qdrant:/qdrant/storage",
        "qdrant/qdrant:latest"
    ])


# In[126]:


qdrant = QdrantClient(url=qdrant_url)
print(f"Qdrant at {qdrant_url}")


# In[127]:


def recreate_collection_if_needed(
    client: QdrantClient,
    collection_name: str,
    vector_dim: int,
):
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"Deleted Old Collection '{collection_name}'")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=vector_dim,
            distance=qmodels.Distance.COSINE,
        ),
    )
    print(f"Created New Collection '{collection_name}' with Dimension:{vector_dim}")

recreate_collection_if_needed(qdrant, qdrant_collection, embed_dim)


# #### Vector Store Upload

# In[128]:


def upload_chunks_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    chunks: List[Chunk],
    embeddings: np.ndarray,
    batch_size: int = 128,
):
    assert len(chunks) == embeddings.shape[0]
    total = len(chunks)
    uploaded = 0

    for i in tqdm(range(0, len(chunks), batch_size), desc="Uploading to Qdrant"):
        batch_chunks = chunks[i : i + batch_size]
        batch_vectors = embeddings[i : i + batch_size]

        points = []
        for c, v in zip(batch_chunks, batch_vectors):
            payload = {
                "chunk_id": c.chunk_id,
                "section": c.section,
                "subsection": c.subsection,
                "block_index": c.block_index,
                "chunk_index": c.chunk_index,
                "text": c.text,
            }
            points.append(
                qmodels.PointStruct(
                    id=c.chunk_id,
                    vector=v.tolist(),
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=collection_name,
            points=points,
        )
        uploaded += len(batch_chunks)

    print(f"Uploaded {uploaded} / {total} Chunks to Qdrant Collection: '{collection_name}'")

upload_chunks_to_qdrant(qdrant, qdrant_collection, chunks, chunk_embeddings)


# In[129]:


info = qdrant.get_collection(qdrant_collection)
print("Qdrant Collection Info:")
print(info)


# #### Vector Store Search Function

# In[130]:


def search_qdrant(
    client: QdrantClient,
    collection_name: str,
    query: str,
    top_k: int = 20,
) -> List[Dict]:
    # 1. Turn the text query into an embedding
    query_vector = embed_query(query)  # should return a 1D numpy array of length 768

    # 2. Use the new universal query_points API
    #    - `query` is the vector (list[float])
    #    - It returns a QueryResponse, whose .points is a list[ScoredPoint]
    response: QueryResponse = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),   # dense vector
        limit=top_k,
        with_payload=True,             # attach payloads to results
        # with_vectors=False by default; set True if you also want stored vectors
    )

    # 3. Normalize output into your desired format
    output: List[Dict] = []
    for p in response.points:
        output.append(
            {
                "id": p.id,
                "score": p.score,
                "payload": p.payload,
                # you could optionally add "vector": p.vector if you call with_vectors=True
            }
        )
    return output


# ### Evaluation

# #### Vector Store Search Function w/ Embedding

# In[131]:


test_query = "What is Application Fraud?"
dense_test_results = search_qdrant(qdrant, qdrant_collection, test_query, top_k=3)
print(f"Retrieval Results for Query: '{test_query}'")
for i, r in enumerate(dense_test_results, start=1):
    print(f"  rank {i}, score={r['score']:.4f}, section={r['payload'].get('section')}, subsection={r['payload'].get('subsection')}")
    print(f"    text snippet='{r['payload']['text'][:150]}...'")


# In[132]:


test_query = "What is the Technology for Detecting Credit Card Frauds?"
dense_test_results = search_qdrant(qdrant, qdrant_collection, test_query, top_k=3)
print(f"Retrieval Results for Query: '{test_query}'")
for i, r in enumerate(dense_test_results, start=1):
    print(f"  rank {i}, score={r['score']:.4f}, section={r['payload'].get('section')}, subsection={r['payload'].get('subsection')}")
    print(f"    text snippet='{r['payload']['text'][:150]}...'")


# In[133]:


test_query = "What is the Key to Minimize Cost of Review?"
dense_test_results = search_qdrant(qdrant, qdrant_collection, test_query, top_k=3)
print(f"Retrieval Results for Query: '{test_query}'")
for i, r in enumerate(dense_test_results, start=1):
    print(f"  rank {i}, score={r['score']:.4f}, section={r['payload'].get('section')}, subsection={r['payload'].get('subsection')}")
    print(f"    text snippet='{r['payload']['text'][:150]}...'")


# #### Vector Store Search Function w/ Embedding & Reranking

# In[134]:


test_query = "What is Application Fraud?"
dense_test_results = search_qdrant(qdrant, qdrant_collection, test_query, top_k=5)
print(f"Retrieval Results for Query: '{test_query}'")
for i, r in enumerate(dense_test_results, start=1):
    print(f"  rank {i}, score={r['score']:.4f}, section={r['payload'].get('section')}, subsection={r['payload'].get('subsection')}")
    print(f"    text snippet='{r['payload']['text'][:150]}...'")
    
    
reranked_test_results = rerank_with_bge_reranker(test_query, dense_test_results, top_k=5)
print(f"\nReranked Retrieval Results for query: '{test_query}'")
for i, r in enumerate(reranked_test_results, start=1):
    print(f"  rerank {i}, rerank_score={r['rerank_score']:.4f}, section={r['payload'].get('section')}, subsection={r['payload'].get('subsection')}")
    print(f"    text snippet='{r['payload']['text'][:150]}...'")


# In[135]:


test_query = "What is the Technology for Detecting Credit Card Frauds?"
dense_test_results = search_qdrant(qdrant, qdrant_collection, test_query, top_k=5)
print(f"Retrieval Results for Query: '{test_query}'")
for i, r in enumerate(dense_test_results, start=1):
    print(f"  rank {i}, score={r['score']:.4f}, section={r['payload'].get('section')}, subsection={r['payload'].get('subsection')}")
    print(f"    text snippet='{r['payload']['text'][:150]}...'")
    
reranked_test_results = rerank_with_bge_reranker(test_query, dense_test_results, top_k=5)
print(f"\nReranked Retrieval Results for query: '{test_query}'")
for i, r in enumerate(reranked_test_results, start=1):
    print(f"  rerank {i}, rerank_score={r['rerank_score']:.4f}, section={r['payload'].get('section')}, subsection={r['payload'].get('subsection')}")
    print(f"    text snippet='{r['payload']['text'][:150]}...'")


# In[136]:


test_query = "What is the Key to Minimize Cost of Review?"
dense_test_results = search_qdrant(qdrant, qdrant_collection, test_query, top_k=5)
print(f"Retrieval Results for Query: '{test_query}'")
for i, r in enumerate(dense_test_results, start=1):
    print(f"  rank {i}, score={r['score']:.4f}, section={r['payload'].get('section')}, subsection={r['payload'].get('subsection')}")
    print(f"    text snippet='{r['payload']['text'][:150]}...'")
    
reranked_test_results = rerank_with_bge_reranker(test_query, dense_test_results, top_k=5)
print(f"\nReranked Retrieval Results for query: '{test_query}'")
for i, r in enumerate(reranked_test_results, start=1):
    print(f"  rerank {i}, rerank_score={r['rerank_score']:.4f}, section={r['payload'].get('section')}, subsection={r['payload'].get('subsection')}")
    print(f"    text snippet='{r['payload']['text'][:150]}...'")


# ## Inference

# ### Import

# In[1]:


import os
from typing import List, Dict, Optional, Any
import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

from qdrant_client import QdrantClient
from qdrant_client.http.models import QueryResponse
from qdrant_client.http import models as qmodels


# ### Config & Variables

# In[2]:


qdrant_url = "http://localhost:6333"
qdrant_collection = "bhatla_credit_fraud"

embed_model_name = "BAAI/bge-base-en-v1.5"
embed_dim = 768

reranker_model_name = "BAAI/bge-reranker-base"


# In[3]:


with open("Bhatla_chunks.json", "r", encoding="utf-8") as f:
    chunk_records: List[Dict] = json.load(f)

chunk_embeddings: np.ndarray = np.load("Bhatla_embeddings.npy")


# In[4]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ### Embedding: BAAI/bge-base-en-v1.5

# #### Embedding Model

# In[5]:


embed_model = SentenceTransformer(embed_model_name, device=device)
print(f"Embedding Model: '{embed_model_name}' on Device: {device}")


# #### Embedding Function

# In[6]:


def embed_texts(texts: List[str], batch_size: int = 32, is_query: bool = False) -> np.ndarray:
    if is_query:
        prefixed = [f"query: {t}" for t in texts]
    else:
        prefixed = [f"passage: {t}" for t in texts]
    embeddings = embed_model.encode(
        prefixed,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings

def embed_query(query: str) -> np.ndarray:
    return embed_texts([query], batch_size=1, is_query=True)[0]


# ### Reranker: BAAI/bge-reranker-base

# #### Reranker Model

# In[7]:


reranker = CrossEncoder(
    reranker_model_name,
    device=device,
    max_length=512,
    trust_remote_code=True,
)
print(f"Reranker Model: '{reranker_model_name}' on Device: {device}")


# #### Reranker Function

# In[8]:


def rerank_with_bge_reranker(
    query: str,
    retrieved_results: List[Dict],
    top_k: Optional[int] = None,
) -> List[Dict]:
    if not retrieved_results:
        return []

    texts = [r["payload"]["text"] for r in retrieved_results]
    pairs = [(query, t) for t in texts]

    scores = reranker.predict(pairs)

    for r, s in zip(retrieved_results, scores):
        r["rerank_score"] = float(s)

    reranked = sorted(retrieved_results, key=lambda x: x["rerank_score"], reverse=True)

    if top_k is not None:
        reranked = reranked[:top_k]

    return reranked


# ### Vector Store: Qdrant

# #### Vector Store Setup and Connection

# In[9]:


import os
import subprocess

os.makedirs("qdrant", exist_ok=True)

if subprocess.run(["docker", "start", "qdrant"]).returncode != 0:
    subprocess.run([
        "docker", "run", "-d",
        "--name", "qdrant",
        "-p", "6333:6333",
        "-v", f"{os.getcwd()}\\qdrant:/qdrant/storage",
        "qdrant/qdrant:latest"
    ])


# In[10]:


qdrant = QdrantClient(url=qdrant_url)
print(f"Qdrant at {qdrant_url}")


# In[12]:


def recreate_collection_if_needed(
    client: QdrantClient,
    collection_name: str,
    vector_dim: int,
):
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"Deleted Old Collection '{collection_name}'")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=vector_dim,
            distance=qmodels.Distance.COSINE,
        ),
    )
    print(f"Created New Collection '{collection_name}' with Dimension:{vector_dim}")

recreate_collection_if_needed(qdrant, qdrant_collection, embed_dim)


# #### Vector Store Upload

# In[13]:


def upload_saved_chunks_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    chunk_records: List[Dict],
    embeddings: np.ndarray,
    batch_size: int = 128,
):
    assert len(chunk_records) == embeddings.shape[0]
    total = len(chunk_records)
    uploaded = 0

    for i in range(0, total, batch_size):
        batch_records = chunk_records[i : i + batch_size]
        batch_vectors = embeddings[i : i + batch_size]

        points = []
        for rec, vec in zip(batch_records, batch_vectors):
            payload = {
                "chunk_id": rec["chunk_id"],
                "section": rec.get("section"),
                "subsection": rec.get("subsection"),
                "block_index": rec.get("block_index"),
                "chunk_index": rec.get("chunk_index"),
                "text": rec["text"],
            }

            points.append(
                qmodels.PointStruct(
                    id=rec["chunk_id"],
                    vector=vec.tolist(),
                    payload=payload,
                )
            )

        client.upsert(collection_name=collection_name, points=points)
        uploaded += len(batch_records)

    print(f"Uploaded {uploaded} / {total} Chunks to Qdrant Collection: '{collection_name}'")


upload_saved_chunks_to_qdrant(qdrant, qdrant_collection, chunk_records, chunk_embeddings)


# In[14]:


info = qdrant.get_collection(qdrant_collection)
print("Qdrant Collection Info:")
print(info)


# #### Vector Store Search Function

# In[15]:


def search_qdrant(
    client: QdrantClient,
    collection_name: str,
    query: str,
    top_k: int = 20,
) -> List[Dict]:
    # 1. Turn the text query into an embedding
    query_vector = embed_query(query)  # should return a 1D numpy array of length 768

    # 2. Use the new universal query_points API
    #    - `query` is the vector (list[float])
    #    - It returns a QueryResponse, whose .points is a list[ScoredPoint]
    response: QueryResponse = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),   # dense vector
        limit=top_k,
        with_payload=True,             # attach payloads to results
        # with_vectors=False by default; set True if you also want stored vectors
    )

    # 3. Normalize output into your desired format
    output: List[Dict] = []
    for p in response.points:
        output.append(
            {
                "id": p.id,
                "score": p.score,
                "payload": p.payload,
                # you could optionally add "vector": p.vector if you call with_vectors=True
            }
        )
    return output


# ### Testing

# In[16]:


def retrieve_chunks(
    query: str,
    top_k: int = 5,
    use_reranker: bool = True,
) -> List[Dict]:
    dense_results = search_qdrant(qdrant, qdrant_collection, query, top_k=top_k)

    if not use_reranker:
        return dense_results

    reranked_results = rerank_with_bge_reranker(query, dense_results, top_k=top_k)
    return reranked_results


# In[17]:


def pretty_print_results(
    query: str,
    results: List[Dict],
    max_chars: int = 200,
):
    print(f"Retrieval Results for Query: '{query}'")
    if not results:
        print("No Result")
        return

    for i, r in enumerate(results, start=1):
        base_score = r.get("score", None)
        rerank_score = r.get("rerank_score", None)
        section = r["payload"].get("section")
        subsection = r["payload"].get("subsection")
        text_snippet = r["payload"]["text"][:max_chars].replace("\n", " ")

        print(f"rank {i}")
        if base_score is not None:
            print(f"    dense_score:   {base_score:.4f}")
        if rerank_score is not None:
            print(f"    rerank_score:  {rerank_score:.4f}")
        print(f"    section:       {section}")
        print(f"    subsection:    {subsection}")
        print(f"    snippet:       '{text_snippet}...'")


# In[18]:


queries = [
    "What is Application Fraud?",
    "What is the Technology for Detecting Credit Card Frauds?",
    "What is the Key to Minimize Cost of Review?",
]

for q in queries:
    results = retrieve_chunks(q, top_k=5, use_reranker=True)
    pretty_print_results(q, results, max_chars=200)

