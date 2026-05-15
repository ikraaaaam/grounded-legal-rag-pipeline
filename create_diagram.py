"""
create_diagram.py — Generates the architecture diagram as a PNG

Run:
    python create_diagram.py

Outputs:
    architecture.png   — PNG diagram for README embedding
    architecture.txt   — ASCII fallback

Uses matplotlib if available, otherwise outputs ASCII only.
"""

from __future__ import annotations

from pathlib import Path

BASE = Path(__file__).resolve().parent

# ── ASCII diagram (always generated) ──────────────────────────────────────

ASCII_DIAGRAM = """
╔══════════════════════════════════════════════════════════════════╗
║            LEGAL AI PIPELINE — SYSTEM ARCHITECTURE              ║
╚══════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────┐
  │   INPUT: PDF / Scanned Document     │
  └─────────────────┬───────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────┐
  │        processor.py                 │
  │   ① Native text extraction (pypdf)  │
  │   ② OCR fallback (Tesseract/PIL)    │
  │   ③ Sliding-window chunking         │
  │   ④ Persist ProcessedDocument JSON  │
  └─────────────────┬───────────────────┘
                    │  List[Chunk]
                    ▼
  ┌─────────────────────────────────────┐
  │        retriever.py                 │
  │   ① Sentence-Transformer embeddings │
  │   ② FAISS IndexFlatIP (cosine sim)  │
  │   ③ Persist index + metadata sidecar│
  └──────────┬──────────────────────────┘
             │
      ┌──────┴──────────────────────────────┐
      │  Query-time retrieval               │
      │  (top-K by cosine similarity)       │
      └──────┬──────────────────────────────┘
             │  List[Evidence]
             ▼
  ┌─────────────────────────────────────┐
  │        generator.py                 │
  │   ① Section-specific queries        │
  │   ② Evidence injected into prompt   │
  │   ③ Claude generates grounded text  │
  │   ④ Hallucination control:          │
  │      → "No clear evidence found"    │
  │        if retrieval is empty        │
  │   ⑤ Per-claim evidence citations    │
  └─────────────────┬───────────────────┘
                    │  MemoOutput
                    ▼
  ┌─────────────────────────────────────┐
  │       Operator Review               │
  │   Human reads & edits the draft     │
  └─────────────────┬───────────────────┘
                    │  (original, edited) pair
                    ▼
  ┌─────────────────────────────────────┐
  │        feedback.py                  │
  │   ① Sentence-level diff (difflib)   │
  │   ② LLM-assisted pattern extraction │
  │   ③ Heuristic terminology swaps     │
  │   ④ Frequency-gated persistence     │
  │   ⑤ Style guidance → next draft     │
  └─────────────────────────────────────┘
                    │
                    └──► Improved future drafts ◄──┐
                         (style_guidance injected   │
                          into system prompt)        │
                                                    │
  ┌─────────────────────────────────────────────────┘
  │  Pattern Learning Loop
  │  Patterns are only applied after MIN_EDITS_TO_LEARN
  │  occurrences, preventing overfitting to a single edit.
  └─────────────────────────────────────────────────────

  ┌─────────────────────────────────────┐
  │    pipeline.py (Orchestrator)       │
  │    Ties all modules end-to-end      │
  │    CLI + importable step functions  │
  └─────────────────────────────────────┘

  ┌─────────────────────────────────────┐
  │    api.py (FastAPI)                 │
  │    REST interface for all steps     │
  │    POST /upload  POST /generate     │
  │    POST /retrieve  POST /feedback   │
  └─────────────────────────────────────┘
"""


def generate_ascii():
    out = BASE / "architecture.txt"
    out.write_text(ASCII_DIAGRAM, encoding="utf-8")
    print(f"✓  ASCII diagram: {out.name}")


def generate_png():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyArrowPatch

        fig, ax = plt.subplots(figsize=(10, 14))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 14)
        ax.axis("off")
        fig.patch.set_facecolor("#0d1117")

        # Color palette
        BLUE   = "#1f6feb"
        GREEN  = "#238636"
        ORANGE = "#d29922"
        PURPLE = "#8957e5"
        RED    = "#da3633"
        GRAY   = "#21262d"
        WHITE  = "#f0f6fc"
        ARROW  = "#58a6ff"

        def box(ax, x, y, w, h, label, sublabels, color):
            rect = mpatches.FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.1",
                linewidth=1.5,
                edgecolor=color,
                facecolor=GRAY,
            )
            ax.add_patch(rect)
            ax.text(x + w/2, y + h - 0.2, label, ha="center", va="top",
                    color=color, fontsize=9, fontweight="bold", fontfamily="monospace")
            for i, sub in enumerate(sublabels):
                ax.text(x + 0.2, y + h - 0.5 - i*0.28, sub,
                        ha="left", va="top", color=WHITE, fontsize=7, fontfamily="monospace")

        def arrow(ax, x1, y1, x2, y2, label=""):
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color=ARROW, lw=1.5))
            if label:
                mx, my = (x1+x2)/2 + 0.1, (y1+y2)/2
                ax.text(mx, my, label, color=ARROW, fontsize=7, fontfamily="monospace")

        # Title
        ax.text(5, 13.7, "LEGAL AI PIPELINE — ARCHITECTURE", ha="center", va="top",
                color=WHITE, fontsize=12, fontweight="bold", fontfamily="monospace")

        # Boxes (x, y, w, h)
        box(ax, 2, 12.2, 6, 0.9, "INPUT — PDF / Scanned Document", [], BLUE)
        arrow(ax, 5, 12.2, 5, 11.6)

        box(ax, 1, 10.2, 8, 1.3, "processor.py -- Ingestion & OCR",
            ["[1] Native extraction (pypdf)", "[2] OCR fallback (Tesseract)", "[3] Sliding-window chunking"], GREEN)
        arrow(ax, 5, 10.2, 5, 9.6, "List[Chunk]")

        box(ax, 1, 8.2, 8, 1.3, "retriever.py -- Embedding & FAISS Index",
            ["[1] sentence-transformers (all-MiniLM-L6-v2)", "[2] FAISS IndexFlatIP cosine similarity", "[3] Persist index + metadata sidecar"], BLUE)
        arrow(ax, 5, 8.2, 5, 7.6, "List[Evidence]")

        box(ax, 1, 5.8, 8, 1.7, "generator.py -- Grounded Memo Generation",
            ["[1] Per-section retrieval queries", "[2] Evidence injected into prompt",
             "[3] Claude writes grounded text", '[4] "No clear evidence found" if empty'], ORANGE)
        arrow(ax, 5, 5.8, 5, 5.2, "MemoOutput")

        box(ax, 2, 4.4, 6, 0.7, "Operator Review", ["Human reads & edits draft"], GRAY)
        arrow(ax, 5, 4.4, 5, 3.8, "(original, edited)")

        box(ax, 1, 2.2, 8, 1.5, "feedback.py -- Pattern Learning Loop",
            ["[1] Sentence-level diff (difflib)", "[2] LLM pattern extraction",
             "[3] Frequency-gated persistence -> style guidance"], PURPLE)

        # Feedback arrow back to generator
        ax.annotate("", xy=(1, 6.7), xytext=(1, 2.9),
                    arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.5,
                                   connectionstyle="arc3,rad=0.0"))
        ax.text(0.1, 4.8, "Improved\nfuture drafts", color=PURPLE, fontsize=7,
                fontfamily="monospace", rotation=90, ha="center")

        # Bottom row
        box(ax, 0.3, 0.2, 4, 1.5, "pipeline.py — Orchestrator",
            ["CLI entry point", "End-to-end workflow", "Step functions for testing"], GREEN)
        box(ax, 5.2, 0.2, 4.5, 1.5, "api.py — FastAPI",
            ["POST /upload", "POST /generate", "POST /retrieve  POST /feedback"], RED)

        plt.tight_layout()
        out = BASE / "architecture.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"✓  PNG diagram: {out.name}")

    except ImportError:
        print("  matplotlib not installed — skipping PNG. ASCII diagram still created.")


if __name__ == "__main__":
    generate_ascii()
    generate_png()
    print("\nDone.  architecture.txt and architecture.png (if matplotlib installed) created.")
