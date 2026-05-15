"""
api.py — FastAPI REST Interface for the Legal AI Pipeline

Run with:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    POST /upload        — Upload and ingest a PDF
    POST /generate      — Generate a grounded memo
    POST /retrieve      — Retrieve evidence for a query
    POST /feedback      — Record an operator edit
    GET  /patterns      — List learned style patterns
    GET  /docs          — List ingested documents
    GET  /health        — Health check
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from config import UPLOADS_DIR
from processor import ingest_pdf, load_processed_doc, list_processed_docs
from retriever import get_retriever
from generator import MemoGenerator
from feedback import record_edit, get_style_guidance, get_feedback_stats, get_all_patterns

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Legal AI Pipeline API",
    description="Grounded legal memo generation with FAISS retrieval, Claude drafting, and operator edit learning.",
    version="1.0.0",
)


# ── Pydantic models ───────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    doc_ids: List[str]
    top_k: int = 6

class FeedbackRequest(BaseModel):
    doc_id: str
    original_text: str
    edited_text: str
    section: Optional[str] = None

class RetrieveRequest(BaseModel):
    query: str
    doc_ids: Optional[List[str]] = None
    top_k: int = 5


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    retriever = get_retriever()
    return {"status": "ok", "index_stats": retriever.stats}


@app.post("/upload", summary="Upload and ingest a PDF")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts any PDF (digital or scanned). The system extracts text via
    native layer or Tesseract OCR fallback, chunks it, embeds with
    sentence-transformers, and stores in FAISS.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=UPLOADS_DIR) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        doc = ingest_pdf(tmp_path)
        retriever = get_retriever()
        added = retriever.add_chunks(doc.chunks)
        return {
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "total_pages": doc.total_pages,
            "chunks": len(doc.chunks),
            "new_indexed": added,
            "extraction": doc.extraction_summary,
        }
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/generate", summary="Generate a grounded legal memo")
def generate_memo(req: GenerateRequest):
    """
    Generates a structured memo grounded in retrieved evidence.
    Unsupported claims are flagged as 'No clear evidence found' — never fabricated.
    """
    retriever = get_retriever()
    if retriever.stats["total_vectors"] == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF first.")
    style_guidance = get_style_guidance()
    generator = MemoGenerator(retriever)
    memo = generator.generate(doc_ids=req.doc_ids, style_guidance=style_guidance, top_k=req.top_k)
    return memo.to_dict()


@app.post("/retrieve", summary="Retrieve evidence for a query")
def retrieve_evidence(req: RetrieveRequest):
    """Semantic search over indexed chunks, returns top-K with similarity scores and provenance."""
    retriever = get_retriever()
    evidence = retriever.retrieve(req.query, top_k=req.top_k, doc_ids=req.doc_ids)
    return [e.to_dict() for e in evidence]


@app.post("/feedback", summary="Record an operator edit")
def submit_feedback(req: FeedbackRequest):
    """
    Diffs original vs. edited draft, extracts reusable patterns via Claude,
    and persists them for future generation improvement.
    """
    return record_edit(
        doc_id=req.doc_id,
        original_text=req.original_text,
        edited_text=req.edited_text,
        section=req.section,
    )


@app.get("/patterns", summary="List all learned style patterns")
def list_patterns():
    return [p.to_dict() for p in get_all_patterns()]


@app.get("/feedback/stats", summary="Feedback loop statistics")
def feedback_stats():
    return get_feedback_stats()


@app.get("/docs", summary="List all ingested documents")
def list_docs():
    return list_processed_docs()


@app.get("/docs/{doc_id}", summary="Get document metadata")
def get_doc(doc_id: str):
    doc = load_processed_doc(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return {
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "total_pages": doc.total_pages,
        "chunks": len(doc.chunks),
        "processed_at": doc.processed_at,
        "extraction_summary": doc.extraction_summary,
    }
