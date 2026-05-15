"""
pipeline.py — End-to-End Orchestrator

This is the single entry point that ties the four subsystems together:

    PDF / Image
        ↓
    OCR + Extraction          (processor.py)
        ↓
    Chunking + Embedding      (retriever.py)
        ↓
    Grounded Memo Generation  (generator.py)
        ↓
    Operator Edit Review      (feedback.py)
        ↓
    Pattern Learning → Improved Future Drafts

Usage:
    # Full pipeline on a PDF
    python pipeline.py --pdf path/to/contract.pdf

    # Generate memo only (document already indexed)
    python pipeline.py --doc-id my_contract --generate-only

    # Record an operator edit so the system learns from it
    python pipeline.py --doc-id my_contract --record-edit \
        --original original.txt --edited edited.txt

    # Show retrieval for a query
    python pipeline.py --doc-id my_contract --query "property tax liens"

    # Run on the built-in sample document (no arguments needed)
    python pipeline.py --demo
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# ── Module imports ──────────────────────────────────────────────────────────
from config import BASE_DIR, PROCESSED_DIR
from processor import ingest_pdf, load_processed_doc
from retriever import Retriever
from generator import MemoGenerator, apply_learned_patterns
from feedback import record_edit, get_style_guidance, get_feedback_stats


# ══════════════════════════════════════════════════════════════════════════
#  Core pipeline steps (importable for FastAPI / tests)
# ══════════════════════════════════════════════════════════════════════════

def step_ingest(pdf_path: Path, doc_id: str | None = None):
    """Step 1 — Ingest a PDF and return a ProcessedDocument."""
    logger.info("━━━ STEP 1 — INGEST ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    doc = ingest_pdf(pdf_path, doc_id=doc_id)
    logger.info(
        "Ingested '%s': %d pages, %d chunks  [native=%d, ocr=%d]",
        doc.filename,
        doc.total_pages,
        len(doc.chunks),
        doc.extraction_summary.get("native_pages", 0),
        doc.extraction_summary.get("ocr_pages", 0),
    )
    return doc


def step_index(retriever: Retriever, doc) -> int:
    """Step 2 — Embed and index the document chunks."""
    logger.info("━━━ STEP 2 — EMBED & INDEX ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    added = retriever.add_chunks(doc.chunks)
    logger.info("Index stats: %s", retriever.stats)
    return added


def step_generate(retriever: Retriever, doc_ids: list[str]) -> dict:
    """Step 3 — Generate a grounded legal memo."""
    logger.info("━━━ STEP 3 — GENERATE MEMO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    style_guidance = get_style_guidance()
    if style_guidance:
        logger.info("Applying %d learned style patterns", style_guidance.count("\n") + 1)

    generator = MemoGenerator(retriever)
    memo = generator.generate(doc_ids=doc_ids, style_guidance=style_guidance)
    return memo


def step_record_edit(doc_id: str, original_path: Path, edited_path: Path, section: str | None = None):
    """Step 4 — Record an operator edit and update the learned patterns."""
    logger.info("━━━ STEP 4 — RECORD EDIT & LEARN ━━━━━━━━━━━━━━━━━━━━━━━")
    original = original_path.read_text(encoding="utf-8")
    edited   = edited_path.read_text(encoding="utf-8")
    result   = record_edit(doc_id, original, edited, section=section)
    logger.info("Edit recorded: %s", result)
    return result


def step_retrieve(retriever: Retriever, query: str, doc_ids: list[str] | None = None) -> list:
    """Standalone retrieval step — useful for inspection and debugging."""
    logger.info("━━━ RETRIEVAL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    evidence = retriever.retrieve(query, doc_ids=doc_ids)
    return evidence


# ══════════════════════════════════════════════════════════════════════════
#  Output helpers
# ══════════════════════════════════════════════════════════════════════════

def print_memo(memo) -> None:
    """Pretty-print the generated memo to stdout."""
    divider = "═" * 70
    print(f"\n{divider}")
    print("  INTERNAL LEGAL MEMO")
    print(f"  Documents: {', '.join(memo.doc_ids)}")
    print(f"  Model: {memo.model}  |  Version: {memo.version}")
    if memo.patterns_applied:
        print(f"  Style patterns applied: {len(memo.patterns_applied)}")
    print(divider)

    for section in memo.sections:
        print(f"\n{'─'*70}")
        print(f"  {section.title.upper()}")
        print(f"{'─'*70}")
        print(section.raw_text)

        if section.claims:
            print(f"\n  [Evidence citations for {len(section.claims)} claims]")
            shown = 0
            for claim in section.claims:
                if claim.evidence and shown < 3:
                    cites = ", ".join(e.citation() for e in claim.evidence)
                    print(f"  • {claim.statement[:120]}…")
                    print(f"    → {cites}")
                    shown += 1

    print(f"\n{divider}\n")


def save_memo(memo, out_path: Path | None = None) -> Path:
    """Save the memo as JSON to disk."""
    if out_path is None:
        doc_id = memo.doc_ids[0] if memo.doc_ids else "memo"
        out_path = BASE_DIR / "data" / "processed" / f"{doc_id}_memo.json"
    out_path.write_text(json.dumps(memo.to_dict(), indent=2))
    logger.info("Memo saved → %s", out_path)
    return out_path


# ══════════════════════════════════════════════════════════════════════════
#  Demo mode — runs the full pipeline on the built-in sample document
# ══════════════════════════════════════════════════════════════════════════

def run_demo() -> None:
    """Full pipeline demo using the sample document."""
    sample_dir = BASE_DIR / "sample_docs"
    sample_pdf = sample_dir / "sample_contract.pdf"

    if not sample_pdf.exists():
        logger.error(
            "Sample PDF not found at %s\n"
            "Run:  python create_sample_pdf.py  to generate it first.",
            sample_pdf,
        )
        sys.exit(1)

    logger.info("Running full pipeline demo on: %s", sample_pdf.name)
    t_start = time.perf_counter()

    # 1. Ingest
    doc = step_ingest(sample_pdf, doc_id="sample_contract")

    # 2. Index
    retriever = Retriever()
    step_index(retriever, doc)

    # 3. Retrieve (inspection)
    logger.info("Sample retrieval for 'payment obligations and deadlines':")
    evidence = step_retrieve(retriever, "payment obligations and deadlines", doc_ids=["sample_contract"])
    for e in evidence[:3]:
        logger.info("  Score=%.3f  %s  %.80s…", e.score, e.citation(), e.text)

    # 4. Generate memo
    memo = step_generate(retriever, doc_ids=["sample_contract"])
    print_memo(memo)
    save_memo(memo)

    # 5. Simulate an operator edit (if sample edits exist)
    orig_edit = sample_dir / "sample_original.txt"
    edit_file  = sample_dir / "sample_edited.txt"
    if orig_edit.exists() and edit_file.exists():
        step_record_edit("sample_contract", orig_edit, edit_file, section="Key Facts")
        stats = get_feedback_stats()
        logger.info("Feedback stats after edit: %s", stats)
    else:
        logger.info("(No sample edit files found — skipping feedback step)")

    logger.info(
        "━━━ Pipeline complete in %.1fs ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        time.perf_counter() - t_start,
    )


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Legal AI Pipeline — grounded memo generation from PDF documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--pdf",           type=Path, help="Path to PDF file to ingest")
    p.add_argument("--doc-id",        type=str,  help="Document ID (default: PDF stem)")
    p.add_argument("--generate-only", action="store_true",
                   help="Skip ingestion; generate memo for already-indexed doc-id")
    p.add_argument("--query",         type=str,  help="Run a retrieval query and print results")
    p.add_argument("--record-edit",   action="store_true",
                   help="Record an operator edit (requires --original and --edited)")
    p.add_argument("--original",      type=Path, help="Path to original memo text file")
    p.add_argument("--edited",        type=Path, help="Path to operator-edited memo text file")
    p.add_argument("--section",       type=str,  help="Section name for the edit record")
    p.add_argument("--save-memo",     type=Path, help="Path to save the generated memo JSON")
    p.add_argument("--demo",          action="store_true",
                   help="Run the full pipeline on the built-in sample document")
    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ── Demo mode ──────────────────────────────────────────────────────────
    if args.demo:
        run_demo()
        return

    # ── Validate args ──────────────────────────────────────────────────────
    if not args.pdf and not args.generate_only and not args.record_edit and not args.query:
        parser.print_help()
        sys.exit(0)

    # ── Ingest ─────────────────────────────────────────────────────────────
    doc      = None
    doc_id   = args.doc_id
    retriever = Retriever()

    if args.pdf:
        doc    = step_ingest(args.pdf, doc_id=doc_id)
        doc_id = doc.doc_id
        step_index(retriever, doc)
    elif args.doc_id and not args.generate_only:
        # Try to load from cache
        doc = load_processed_doc(args.doc_id)
        if doc:
            logger.info("Loaded cached document: %s", args.doc_id)
        else:
            logger.error("No processed document found for doc_id='%s'. Run with --pdf first.", args.doc_id)
            sys.exit(1)

    # ── Retrieval ──────────────────────────────────────────────────────────
    if args.query and doc_id:
        evidence = step_retrieve(retriever, args.query, doc_ids=[doc_id] if doc_id else None)
        print(f"\nRetrieval results for: '{args.query}'\n{'─'*60}")
        if not evidence:
            print("  No evidence found above the similarity threshold.")
        for i, e in enumerate(evidence, 1):
            print(f"\n[{i}] {e.citation()}  (score={e.score:.4f})")
            print(f"    {e.text[:300]}{'…' if len(e.text) > 300 else ''}")
        return

    # ── Generate memo ──────────────────────────────────────────────────────
    if args.generate_only or args.pdf:
        if not doc_id:
            logger.error("--doc-id required when using --generate-only")
            sys.exit(1)
        memo = step_generate(retriever, doc_ids=[doc_id])
        print_memo(memo)
        saved = save_memo(memo, out_path=args.save_memo)
        print(f"Memo saved to: {saved}")

    # ── Record edit ────────────────────────────────────────────────────────
    if args.record_edit:
        if not (args.original and args.edited and args.doc_id):
            logger.error("--record-edit requires --doc-id, --original, and --edited")
            sys.exit(1)
        result = step_record_edit(args.doc_id, args.original, args.edited, section=args.section)
        print(f"\nEdit recorded: {json.dumps(result, indent=2)}")
        print(f"Feedback stats: {json.dumps(get_feedback_stats(), indent=2)}")


if __name__ == "__main__":
    main()
