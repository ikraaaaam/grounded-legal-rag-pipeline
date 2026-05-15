"""
Central configuration for the Legal AI Pipeline.
All tuneable constants live here — no magic numbers scattered across modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent   # flat layout: all files in same dir
DATA_DIR        = BASE_DIR / "data"
UPLOADS_DIR     = DATA_DIR / "uploads"
PROCESSED_DIR   = DATA_DIR / "processed"
VECTOR_DIR      = DATA_DIR / "vector_store"
FEEDBACK_DIR    = DATA_DIR / "feedback"
SAMPLE_DIR      = BASE_DIR / "sample_docs"

for d in [UPLOADS_DIR, PROCESSED_DIR, VECTOR_DIR, FEEDBACK_DIR, SAMPLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── LLM ───────────────────────────────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
LLM_MODEL           = "llama-3.3-70b-versatile"
MAX_TOKENS          = 2048

# ── Embedding / Retrieval ─────────────────────────────────────────────────
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"   # fast, 384-dim, good enough for legal
CHUNK_SIZE          = 400                   # tokens per chunk
CHUNK_OVERLAP       = 80                    # overlap to preserve context at boundaries
TOP_K               = 6                     # passages retrieved per memo section
MIN_SCORE           = 0.25                  # cosine-similarity floor (filter noise)

# ── FAISS index file ──────────────────────────────────────────────────────
FAISS_INDEX_PATH    = VECTOR_DIR / "index.faiss"
METADATA_PATH       = VECTOR_DIR / "metadata.json"

# ── Feedback / Learning ───────────────────────────────────────────────────
FEEDBACK_DB_PATH    = FEEDBACK_DIR / "feedback.json"
PATTERNS_PATH       = FEEDBACK_DIR / "learned_patterns.json"
MIN_EDITS_TO_LEARN  = 2   # need at least N examples before applying a pattern

# ── OCR ───────────────────────────────────────────────────────────────────
OCR_DPI             = 300
OCR_LANG            = "eng"
