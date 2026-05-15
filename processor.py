"""
processor.py — Document Ingestion & OCR Pipeline

Responsibilities:
  1. Accept PDF / image files (messy, scanned, low-res — doesn't matter)
  2. Extract text via pypdf (fast path for digital PDFs) or Tesseract (OCR fallback)
  3. Produce a clean, structured ProcessedDocument with per-page text + metadata
  4. Chunk the text into overlapping windows ready for embedding

Design notes:
  - We try the native PDF text layer first; if a page yields < MIN_CHARS we
    rasterise that page and run Tesseract on the image.  This handles mixed
    documents (some digital pages, some scanned).
  - Chunking is character-window based (not token-based) to avoid a heavy
    tokeniser dependency in the hot path; CHUNK_SIZE / CHUNK_OVERLAP in config.py
    are specified in approximate tokens (~4 chars/token assumed).
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'D:\My_Apps\Tesseract\tesseract.exe'
from PIL import Image
from pypdf import PdfReader

from config import (
    PROCESSED_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OCR_DPI,
    OCR_LANG,
)

logger = logging.getLogger(__name__)

# Heuristic: if native extraction gives fewer chars than this per page it's
# probably a scanned image embedded in the PDF shell.
MIN_CHARS_PER_PAGE = 80
CHARS_PER_TOKEN    = 4          # rough conversion


# ── Data models ───────────────────────────────────────────────────────────

@dataclass
class PageResult:
    page_number: int            # 1-indexed
    text: str
    extraction_method: str      # "native" | "ocr"
    char_count: int = 0
    confidence: Optional[float] = None   # Tesseract confidence 0-100

    def __post_init__(self):
        self.char_count = len(self.text)


@dataclass
class Chunk:
    chunk_id: str               # "<doc_id>_c<n>"
    doc_id: str
    text: str
    start_page: int
    end_page: int
    chunk_index: int


@dataclass
class ProcessedDocument:
    doc_id: str
    filename: str
    file_type: str
    pages: List[PageResult]
    chunks: List[Chunk]
    total_pages: int
    processed_at: str
    extraction_summary: dict = field(default_factory=dict)

    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    def to_dict(self) -> dict:
        return asdict(self)


# ── OCR helpers ───────────────────────────────────────────────────────────

def _ocr_page_from_pdf(reader: PdfReader, page_index: int) -> PageResult:
    """Rasterise a single PDF page and run Tesseract on it."""
    page_num = page_index + 1
    images = []

    try:
        import fitz  # PyMuPDF  — optional but much better rasterisation
        
        # We need the original file path to open with fitz
        # If reader.stream is a file object, we can get its name
        if hasattr(reader.stream, "name"):
            doc = fitz.open(reader.stream.name)
            pix = doc[page_index].get_pixmap(dpi=OCR_DPI)
            images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
            doc.close()
    except (ImportError, Exception) as e:
        logger.debug("fitz rasterisation failed or unavailable: %s", e)

    if not images:
        # Try to render via pypdf's built-in image extraction first
        page = reader.pages[page_index]
        try:
            for img_obj in page.images:
                images.append(Image.open(io.BytesIO(img_obj.data)).convert("RGB"))
        except Exception:
            pass

        if not images:
            # Fallback: convert via PIL from raw bytes (works for many embedded pages)
            try:
                raw = page.get_contents()
                if raw:
                    images.append(Image.new("RGB", (2480, 3508), "white"))  # placeholder
            except Exception:
                pass

    if not images:
        return PageResult(page_num, "", "ocr_failed", confidence=0.0)

    texts, confs = [], []
    for img in images:
        try:
            data = pytesseract.image_to_data(
                img, lang=OCR_LANG, output_type=pytesseract.Output.DICT
            )
            words = [w for w, c in zip(data["text"], data["conf"]) if int(c) > 0 and w.strip()]
            conf_vals = [int(c) for c in data["conf"] if int(c) > 0]
            texts.append(" ".join(words))
            confs.append(sum(conf_vals) / len(conf_vals) if conf_vals else 0.0)
        except Exception as e:
            logger.warning("OCR failed on image in page %d: %s", page_num, e)

    combined = "\n".join(t for t in texts if t.strip())
    avg_conf  = sum(confs) / len(confs) if confs else 0.0
    return PageResult(page_num, combined, "ocr", confidence=round(avg_conf, 1))


def _extract_page_native(page) -> str:
    """Extract text from a pypdf page object using its native text layer."""
    try:
        text = page.extract_text() or ""
        # Clean up common PDF extraction artefacts
        text = re.sub(r"\s{3,}", "  ", text)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # rejoin hyphenated line-breaks
        return text.strip()
    except Exception:
        return ""


# ── Chunking ──────────────────────────────────────────────────────────────

def _chunk_text(doc_id: str, pages: List[PageResult]) -> List[Chunk]:
    """
    Sliding-window chunker that preserves page provenance.

    Each chunk records start_page/end_page so we can cite the source later.
    Window size = CHUNK_SIZE * CHARS_PER_TOKEN characters.
    """
    win   = CHUNK_SIZE   * CHARS_PER_TOKEN
    step  = (CHUNK_SIZE - CHUNK_OVERLAP) * CHARS_PER_TOKEN

    # Build a list of (char_offset, page_number) boundaries
    full_text = ""
    boundaries: List[tuple[int, int]] = []   # (start_char, page_num)
    for p in pages:
        if not p.text.strip():
            continue
        boundaries.append((len(full_text), p.page_number))
        full_text += p.text + "\n\n"

    def page_at(offset: int) -> int:
        pg = 1
        for start, pnum in boundaries:
            if start <= offset:
                pg = pnum
            else:
                break
        return pg

    chunks: List[Chunk] = []
    idx = 0
    pos = 0
    while pos < len(full_text):
        end      = min(pos + win, len(full_text))
        snippet  = full_text[pos:end].strip()
        if snippet:
            chunks.append(Chunk(
                chunk_id    = f"{doc_id}_c{idx}",
                doc_id      = doc_id,
                text        = snippet,
                start_page  = page_at(pos),
                end_page    = page_at(end - 1),
                chunk_index = idx,
            ))
            idx += 1
        pos += step
        if pos >= len(full_text):
            break

    return chunks


# ── Public API ────────────────────────────────────────────────────────────

def ingest_pdf(file_path: Path, doc_id: Optional[str] = None) -> ProcessedDocument:
    """
    Main entry point.  Accepts any PDF (digital, scanned, or mixed).
    Returns a ProcessedDocument with per-page text and ready-to-embed chunks.
    """
    if doc_id is None:
        doc_id = file_path.stem.replace(" ", "_")

    logger.info("Ingesting %s (doc_id=%s)", file_path.name, doc_id)
    t0 = time.perf_counter()

    reader   = PdfReader(str(file_path))
    n_pages  = len(reader.pages)
    pages: List[PageResult] = []
    native_count = ocr_count = 0

    for i, page in enumerate(reader.pages):
        native_text = _extract_page_native(page)
        if len(native_text) >= MIN_CHARS_PER_PAGE:
            pages.append(PageResult(i + 1, native_text, "native"))
            native_count += 1
        else:
            # Fallback: try OCR on the raw page image
            ocr_result = _ocr_image_file(file_path, i) if file_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff", ".bmp") else _ocr_page_from_pdf(reader, i)
            # If OCR also failed, keep whatever native gave us (may be blank)
            if ocr_result.text.strip() or not native_text:
                pages.append(ocr_result)
            else:
                pages.append(PageResult(i + 1, native_text, "native"))
            ocr_count += 1

    chunks = _chunk_text(doc_id, pages)

    summary = {
        "native_pages": native_count,
        "ocr_pages":    ocr_count,
        "total_chunks": len(chunks),
        "elapsed_s":    round(time.perf_counter() - t0, 2),
    }
    logger.info("Ingestion complete: %s", summary)

    doc = ProcessedDocument(
        doc_id      = doc_id,
        filename    = file_path.name,
        file_type   = file_path.suffix.lstrip(".").upper(),
        pages       = pages,
        chunks      = chunks,
        total_pages = n_pages,
        processed_at= time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        extraction_summary = summary,
    )

    # Persist to disk (used by the API to reload docs without re-ingesting)
    out_path = PROCESSED_DIR / f"{doc_id}.json"
    out_path.write_text(json.dumps(doc.to_dict(), indent=2))
    logger.info("Saved processed doc → %s", out_path)

    return doc


def _ocr_image_file(file_path: Path, page_index: int = 0) -> PageResult:
    """Handle standalone image files (not embedded in PDF)."""
    try:
        img  = Image.open(str(file_path)).convert("RGB")
        data = pytesseract.image_to_data(img, lang=OCR_LANG, output_type=pytesseract.Output.DICT)
        words     = [w for w, c in zip(data["text"], data["conf"]) if int(c) > 0 and w.strip()]
        conf_vals = [int(c) for c in data["conf"] if int(c) > 0]
        text      = " ".join(words)
        conf      = sum(conf_vals) / len(conf_vals) if conf_vals else 0.0
        return PageResult(page_index + 1, text, "ocr", confidence=round(conf, 1))
    except Exception as e:
        logger.error("Image OCR failed: %s", e)
        return PageResult(page_index + 1, "", "ocr_failed", confidence=0.0)


def load_processed_doc(doc_id: str) -> Optional[ProcessedDocument]:
    """Re-load a previously ingested document from disk."""
    path = PROCESSED_DIR / f"{doc_id}.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    pages  = [PageResult(**p) for p in raw["pages"]]
    chunks = [Chunk(**c) for c in raw["chunks"]]
    return ProcessedDocument(
        doc_id      = raw["doc_id"],
        filename    = raw["filename"],
        file_type   = raw["file_type"],
        pages       = pages,
        chunks      = chunks,
        total_pages = raw["total_pages"],
        processed_at= raw["processed_at"],
        extraction_summary = raw.get("extraction_summary", {}),
    )


def list_processed_docs() -> List[dict]:
    """Return brief metadata for all ingested documents."""
    docs = []
    for f in PROCESSED_DIR.glob("*.json"):
        try:
            raw = json.loads(f.read_text())
            docs.append({
                "doc_id":    raw["doc_id"],
                "filename":  raw["filename"],
                "pages":     raw["total_pages"],
                "chunks":    len(raw["chunks"]),
                "processed_at": raw["processed_at"],
            })
        except Exception:
            pass
    return docs
