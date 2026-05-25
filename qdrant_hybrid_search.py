"""
Qdrant Hybrid Search Starter
=============================
Demonstrates:
  - Dense + Sparse (BM25) hybrid search via Qdrant named vectors
  - PDF ingestion with source metadata
  - Multiple collections
  - Metadata filtering
  - Query: single collection / across multiple collections

Dependencies (install once):
    pip install qdrant-client fastembed pypdf

fastembed supplies:
  - Dense  : BAAI/bge-small-en-v1.5   (384-dim)
  - Sparse : Qdrant/bm25               (BM25 sparse vectors)

Run Qdrant locally (Docker):
    docker run -p 6333:6333 qdrant/qdrant
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastembed import SparseTextEmbedding, TextEmbedding
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    Prefetch,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

DENSE_MODEL  = "BAAI/bge-small-en-v1.5"   # 384-dim
SPARSE_MODEL = "Qdrant/bm25"              # BM25 sparse

DENSE_DIM    = 384
DENSE_NAME   = "dense"
SPARSE_NAME  = "sparse"

CHUNK_SIZE   = 400   # characters per chunk
CHUNK_OVERLAP = 80


# ──────────────────────────────────────────────────────────────────────────────
# Embedders (lazy singletons)
# ──────────────────────────────────────────────────────────────────────────────

_dense_embedder:  TextEmbedding | None = None
_sparse_embedder: SparseTextEmbedding | None = None


def get_dense_embedder() -> TextEmbedding:
    global _dense_embedder
    if _dense_embedder is None:
        print(f"[embedder] Loading dense model: {DENSE_MODEL}")
        _dense_embedder = TextEmbedding(DENSE_MODEL)
    return _dense_embedder


def get_sparse_embedder() -> SparseTextEmbedding:
    global _sparse_embedder
    if _sparse_embedder is None:
        print(f"[embedder] Loading sparse model: {SPARSE_MODEL}")
        _sparse_embedder = SparseTextEmbedding(SPARSE_MODEL)
    return _sparse_embedder


def embed_dense(texts: list[str]) -> list[list[float]]:
    """Return a list of dense vectors."""
    return [v.tolist() for v in get_dense_embedder().embed(texts)]


def embed_sparse(texts: list[str]) -> list[dict]:
    """Return a list of {indices, values} dicts for BM25 sparse vectors."""
    results = []
    for sv in get_sparse_embedder().embed(texts):
        results.append({"indices": sv.indices.tolist(), "values": sv.values.tolist()})
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A text chunk ready for upsert."""
    text:       str
    source:     str          # PDF filename (used as metadata)
    page:       int
    chunk_idx:  int
    extra_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def payload(self) -> dict:
        return {
            "text":      self.text,
            "source":    self.source,
            "page":      self.page,
            "chunk_idx": self.chunk_idx,
            **self.extra_meta,
        }


# ──────────────────────────────────────────────────────────────────────────────
# PDF helpers
# ──────────────────────────────────────────────────────────────────────────────

def pdf_to_chunks(pdf_path: str | Path, extra_meta: dict | None = None) -> list[Chunk]:
    """
    Extract text from a PDF and split into overlapping chunks.
    The PDF filename (stem) is stored as `source` in metadata.
    """
    path   = Path(pdf_path)
    source = path.stem                          # e.g. "attention_is_all_you_need"
    meta   = extra_meta or {}
    reader = PdfReader(str(path))
    chunks: list[Chunk] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        # Slide a window over the page text
        start     = 0
        chunk_idx = 0
        while start < len(text):
            end  = start + CHUNK_SIZE
            body = text[start:end].strip()
            if body:
                chunks.append(Chunk(
                    text=body,
                    source=source,
                    page=page_num,
                    chunk_idx=chunk_idx,
                    extra_meta=meta,
                ))
                chunk_idx += 1
            start = end - CHUNK_OVERLAP

    print(f"[pdf] '{source}' → {len(chunks)} chunks from {len(reader.pages)} pages")
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Collection management
# ──────────────────────────────────────────────────────────────────────────────

def create_hybrid_collection(client: QdrantClient, collection_name: str) -> None:
    """
    Create a Qdrant collection with:
      - 'dense'  : cosine-similarity dense vectors
      - 'sparse' : BM25 sparse vectors (dot-product)
    Skips creation if collection already exists.
    """
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        print(f"[collection] '{collection_name}' already exists — skipping creation")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_NAME: VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            ),
        },
    )
    print(f"[collection] Created '{collection_name}'")


# ──────────────────────────────────────────────────────────────────────────────
# Ingestion
# ──────────────────────────────────────────────────────────────────────────────

def ingest_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[Chunk],
    batch_size: int = 32,
) -> None:
    """
    Embed and upsert chunks into a Qdrant collection in batches.
    Each point stores both dense + sparse vectors plus the payload.
    """
    from qdrant_client.http.models import PointStruct, SparseVector

    total = len(chunks)
    print(f"[ingest] Upserting {total} chunks into '{collection_name}' …")

    for i in range(0, total, batch_size):
        batch  = chunks[i : i + batch_size]
        texts  = [c.text for c in batch]

        dense_vecs  = embed_dense(texts)
        sparse_vecs = embed_sparse(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    DENSE_NAME:  dense_vecs[j],
                    SPARSE_NAME: SparseVector(
                        indices=sparse_vecs[j]["indices"],
                        values=sparse_vecs[j]["values"],
                    ),
                },
                payload=batch[j].payload,
            )
            for j in range(len(batch))
        ]

        client.upsert(collection_name=collection_name, points=points)
        print(f"  ↳ batch {i // batch_size + 1}: upserted {len(points)} points")

    print(f"[ingest] Done — {total} chunks in '{collection_name}'")


def ingest_pdf(
    client: QdrantClient,
    collection_name: str,
    pdf_path: str | Path,
    extra_meta: dict | None = None,
) -> None:
    """High-level helper: parse PDF → chunk → embed → upsert."""
    create_hybrid_collection(client, collection_name)
    chunks = pdf_to_chunks(pdf_path, extra_meta)
    ingest_chunks(client, collection_name, chunks)


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid search
# ──────────────────────────────────────────────────────────────────────────────

def hybrid_search(
    client: QdrantClient,
    collection_name: str,
    query: str,
    top_k: int = 5,
    metadata_filter: Filter | None = None,
) -> list[dict]:
    """
    Hybrid search using Qdrant's query API with Reciprocal Rank Fusion (RRF).

    Steps:
      1. Prefetch top-k dense candidates (cosine similarity)
      2. Prefetch top-k sparse candidates (BM25 dot-product)
      3. Fuse via RRF → return top_k results

    An optional Qdrant Filter can be passed for metadata filtering.
    """
    from qdrant_client.http.models import FusionQuery, SparseVector

    dense_vec  = embed_dense([query])[0]
    sparse_vec = embed_sparse([query])[0]

    prefetch = [
        # Dense branch
        Prefetch(
            query=dense_vec,
            using=DENSE_NAME,
            limit=top_k * 3,
            filter=metadata_filter,
        ),
        # Sparse / BM25 branch
        Prefetch(
            query=SparseVector(
                indices=sparse_vec["indices"],
                values=sparse_vec["values"],
            ),
            using=SPARSE_NAME,
            limit=top_k * 3,
            filter=metadata_filter,
        ),
    ]

    results = client.query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        query=FusionQuery(fusion="rrf"),   # Reciprocal Rank Fusion
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "score":  hit.score,
            "source": hit.payload.get("source"),
            "page":   hit.payload.get("page"),
            "text":   hit.payload.get("text", "")[:200] + "…",
            "meta":   {k: v for k, v in hit.payload.items()
                       if k not in ("text", "source", "page", "chunk_idx")},
        }
        for hit in results.points
    ]


def hybrid_search_multi_collection(
    client: QdrantClient,
    collection_names: list[str],
    query: str,
    top_k_per_collection: int = 3,
    metadata_filter: Filter | None = None,
) -> dict[str, list[dict]]:
    """
    Run hybrid_search against multiple collections and return results keyed
    by collection name. Useful for cross-collection queries.
    """
    return {
        name: hybrid_search(client, name, query, top_k_per_collection, metadata_filter)
        for name in collection_names
    }


# ──────────────────────────────────────────────────────────────────────────────
# Metadata filter helpers  (convenience wrappers around Qdrant models)
# ──────────────────────────────────────────────────────────────────────────────

def filter_by_source(source: str) -> Filter:
    """Match a single PDF source (filename stem)."""
    return Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])


def filter_by_sources(sources: list[str]) -> Filter:
    """Match any of the given PDF sources."""
    return Filter(must=[FieldCondition(key="source", match=MatchAny(any=sources))])


def filter_by_page_range(start: int, end: int) -> Filter:
    """Match pages within [start, end] (inclusive)."""
    from qdrant_client.http.models import Range
    return Filter(must=[FieldCondition(key="page", range=Range(gte=start, lte=end))])


def combined_filter(*filters: Filter) -> Filter:
    """AND-combine multiple filters."""
    must_conditions = []
    for f in filters:
        must_conditions.extend(f.must or [])
    return Filter(must=must_conditions)


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────────────────────────────────────

def print_results(results: list[dict], label: str = "Results") -> None:
    print(f"\n{'═' * 60}")
    print(f"  {label}  ({len(results)} hits)")
    print(f"{'═' * 60}")
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] score={r['score']:.4f}  source={r['source']}  page={r['page']}")
        if r["meta"]:
            print(f"      meta={r['meta']}")
        print(f"      {r['text']}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Demo  (replace PDF paths with your own files)
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Connect to Qdrant (local Docker) ──────────────────────────────────
    client = QdrantClient(host="localhost", port=6333)
    # For in-memory testing (no Docker needed):
    # client = QdrantClient(":memory:")

    # ── 2. Ingest PDFs into two separate collections ──────────────────────────
    #
    # Collection "ai_papers"  → two AI-related PDFs
    # Collection "law_docs"   → two legal PDFs
    #
    # Replace the paths below with real PDF files on your machine.
    # The PDF filename stem becomes the `source` metadata field automatically.

    ingest_pdf(client, "ai_papers",  "attention_is_all_you_need.pdf",
               extra_meta={"domain": "nlp", "year": 2017})

    ingest_pdf(client, "ai_papers",  "bert_pretraining.pdf",
               extra_meta={"domain": "nlp", "year": 2019})

    ingest_pdf(client, "law_docs",   "gdpr_full_text.pdf",
               extra_meta={"domain": "regulation", "jurisdiction": "EU"})

    ingest_pdf(client, "law_docs",   "california_privacy_act.pdf",
               extra_meta={"domain": "regulation", "jurisdiction": "US"})

    # ── 3. Query: single collection, no filter ────────────────────────────────
    q = "How does self-attention mechanism work?"

    results = hybrid_search(client, "ai_papers", q, top_k=4)
    print_results(results, f"[ai_papers] '{q}'")

    # ── 4. Query: single collection + filter by ONE source PDF ────────────────
    results = hybrid_search(
        client, "ai_papers", q,
        top_k=3,
        metadata_filter=filter_by_source("attention_is_all_you_need"),
    )
    print_results(results, f"[ai_papers | source=attention_is_all_you_need] '{q}'")

    # ── 5. Query: single collection + filter by MULTIPLE source PDFs ──────────
    results = hybrid_search(
        client, "ai_papers", q,
        top_k=5,
        metadata_filter=filter_by_sources(["attention_is_all_you_need", "bert_pretraining"]),
    )
    print_results(results, f"[ai_papers | 2 sources] '{q}'")

    # ── 6. Query: single collection + combined filter (source + page range) ───
    results = hybrid_search(
        client, "ai_papers", q,
        top_k=3,
        metadata_filter=combined_filter(
            filter_by_source("bert_pretraining"),
            filter_by_page_range(1, 5),
        ),
    )
    print_results(results, f"[ai_papers | bert pages 1-5] '{q}'")

    # ── 7. Query: MULTIPLE collections (cross-collection search) ──────────────
    q2 = "data privacy and user consent"

    multi = hybrid_search_multi_collection(
        client,
        collection_names=["ai_papers", "law_docs"],
        query=q2,
        top_k_per_collection=3,
    )
    for col, hits in multi.items():
        print_results(hits, f"[{col}] '{q2}'")

    # ── 8. Cross-collection with metadata filter ──────────────────────────────
    multi_filtered = hybrid_search_multi_collection(
        client,
        collection_names=["ai_papers", "law_docs"],
        query=q2,
        top_k_per_collection=3,
        metadata_filter=Filter(
            must=[FieldCondition(key="domain", match=MatchValue(value="regulation"))]
        ),
    )
    for col, hits in multi_filtered.items():
        print_results(hits, f"[{col} | domain=regulation] '{q2}'")


if __name__ == "__main__":
    main()
