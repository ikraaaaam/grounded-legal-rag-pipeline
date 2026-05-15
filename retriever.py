"""
retriever.py — Embedding & Grounded Retrieval Layer

Responsibilities:
  1. Encode document chunks with a sentence-transformer model
  2. Store / load a FAISS index (persisted to disk across restarts)
  3. Retrieve top-K evidence passages for a query with similarity scores
  4. Return structured Evidence objects that carry full provenance

Design notes:
  - We use cosine similarity (FAISS IndexFlatIP on normalised vectors).
  - The metadata sidecar (metadata.json) maps FAISS integer IDs → Chunk metadata
    so we can reconstruct full provenance without storing tensors in RAM.
  - The retriever is stateless once loaded; add_chunks() is the only mutating op.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    TOP_K,
    MIN_SCORE,
)
from processor import Chunk

logger = logging.getLogger(__name__)


# ── Evidence model ────────────────────────────────────────────────────────

@dataclass
class Evidence:
    chunk_id:   str
    doc_id:     str
    text:       str
    start_page: int
    end_page:   int
    score:      float           # cosine similarity  0..1

    def citation(self) -> str:
        if self.start_page == self.end_page:
            return f"[{self.doc_id}, p.{self.start_page}]"
        return f"[{self.doc_id}, pp.{self.start_page}–{self.end_page}]"

    def to_dict(self) -> dict:
        return asdict(self)


# ── Retriever ─────────────────────────────────────────────────────────────

class Retriever:
    """
    FAISS-backed dense retriever with cosine similarity.

    Usage:
        r = Retriever()
        r.add_chunks(processed_doc.chunks)
        evidence = r.retrieve("property tax liens", top_k=5)
    """

    def __init__(self):
        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.IndexFlatIP]   = None
        self._meta:  List[dict] = []   # parallel list to FAISS rows
        self._dim:   int = 384         # all-MiniLM-L6-v2 output dim
        self._load()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def _encode(self, texts: List[str]) -> np.ndarray:
        model = self._get_model()
        vecs  = model.encode(texts, batch_size=32, show_progress_bar=False,
                             normalize_embeddings=True)
        return vecs.astype("float32")

    def _load(self):
        """Load persisted index + metadata from disk (if they exist)."""
        if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists():
            try:
                self._index = faiss.read_index(str(FAISS_INDEX_PATH))
                self._meta  = json.loads(METADATA_PATH.read_text())
                logger.info("Loaded FAISS index (%d vectors)", self._index.ntotal)
                return
            except Exception as e:
                logger.warning("Failed to load index, rebuilding: %s", e)
        self._index = faiss.IndexFlatIP(self._dim)
        self._meta  = []

    def _save(self):
        faiss.write_index(self._index, str(FAISS_INDEX_PATH))
        METADATA_PATH.write_text(json.dumps(self._meta, indent=2))
        logger.info("Saved FAISS index (%d vectors)", self._index.ntotal)

    # ── Public API ────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """
        Embed and add chunks to the index.
        Skips chunks already in the index (matched by chunk_id).
        Returns the number of newly added chunks.
        """
        existing_ids = {m["chunk_id"] for m in self._meta}
        new_chunks   = [c for c in chunks if c.chunk_id not in existing_ids]
        if not new_chunks:
            logger.info("All chunks already indexed, skipping.")
            return 0

        texts = [c.text for c in new_chunks]
        vecs  = self._encode(texts)
        self._index.add(vecs)
        for c in new_chunks:
            self._meta.append({
                "chunk_id":   c.chunk_id,
                "doc_id":     c.doc_id,
                "text":       c.text,
                "start_page": c.start_page,
                "end_page":   c.end_page,
            })
        self._save()
        logger.info("Added %d new chunks (index total: %d)", len(new_chunks), self._index.ntotal)
        return len(new_chunks)

    def retrieve(self, query: str, top_k: int = TOP_K, min_score: float = MIN_SCORE,
                 doc_ids: Optional[List[str]] = None) -> List[Evidence]:
        """
        Retrieve the top-K most relevant chunks for a query.

        Args:
            query:     Natural-language query string
            top_k:     Maximum number of results
            min_score: Minimum cosine similarity threshold
            doc_ids:   If set, restrict results to these document IDs

        Returns:
            List of Evidence objects, sorted by descending score.
        """
        if self._index.ntotal == 0:
            logger.warning("Index is empty — no evidence to retrieve.")
            return []

        q_vec = self._encode([query])
        k     = min(top_k * 3, self._index.ntotal)   # over-fetch for filtering
        scores, idxs = self._index.search(q_vec, k)

        results: List[Evidence] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or float(score) < min_score:
                continue
            m = self._meta[idx]
            if doc_ids and m["doc_id"] not in doc_ids:
                continue
            results.append(Evidence(
                chunk_id   = m["chunk_id"],
                doc_id     = m["doc_id"],
                text       = m["text"],
                start_page = m["start_page"],
                end_page   = m["end_page"],
                score      = round(float(score), 4),
            ))
            if len(results) >= top_k:
                break

        logger.info("Retrieved %d evidence chunks for query: %.60s…", len(results), query)
        return results

    def remove_doc(self, doc_id: str):
        """
        Remove all chunks belonging to a document.
        FAISS IndexFlatIP does not support in-place deletion, so we rebuild.
        """
        keep = [(i, m) for i, m in enumerate(self._meta) if m["doc_id"] != doc_id]
        if len(keep) == len(self._meta):
            return   # nothing to remove

        new_index = faiss.IndexFlatIP(self._dim)
        new_meta  = []
        if keep:
            keep_idxs = [i for i, _ in keep]
            all_vecs  = faiss.rev_swig_ptr(self._index.get_xb(), self._index.ntotal * self._dim)
            all_vecs  = np.frombuffer(all_vecs, dtype="float32").reshape(self._index.ntotal, self._dim)
            kept_vecs = all_vecs[keep_idxs]
            new_index.add(kept_vecs)
            new_meta  = [m for _, m in keep]

        self._index = new_index
        self._meta  = new_meta
        self._save()
        logger.info("Removed doc %s from index (%d chunks remaining)", doc_id, len(new_meta))

    @property
    def stats(self) -> dict:
        doc_ids = list({m["doc_id"] for m in self._meta})
        return {
            "total_vectors": self._index.ntotal if self._index else 0,
            "documents":     len(doc_ids),
            "doc_ids":       doc_ids,
        }


# Singleton retriever shared across the FastAPI app
_retriever: Optional[Retriever] = None

def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
