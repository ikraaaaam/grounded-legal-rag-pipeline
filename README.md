# Legal AI Pipeline

> **Grounded legal memo generation** from raw PDF documents — combining OCR extraction, dense retrieval, Llama 3-powered drafting, and a real operator-edit learning loop.

---

## System Architecture

```
PDF / Image
    ↓
OCR + Text Extraction      (processor.py)
    ↓
Chunking
    ↓
Sentence-Transformer Embeddings
    ↓
FAISS Retrieval            (retriever.py)
    ↓
Grounded Memo Generation   (generator.py)
    — evidence injected per section
    — "No clear evidence found" if retrieval empty
    ↓
Operator Edit Review
    ↓
Pattern Learning           (feedback.py)
    ↓
Improved Future Drafts  ←──── (loop)
```

See [`architecture.txt`](architecture.txt) for the detailed ASCII diagram, or run `python create_diagram.py` to generate `architecture.png`.

---

## Key Features

| Feature | Detail |
|---|---|
| **Hybrid OCR** | Native pypdf layer first; Tesseract fallback for scanned/mixed PDFs |
| **FAISS dense retrieval** | Cosine similarity over sentence-transformer embeddings; persisted across restarts |
| **Grounded generation** | Every memo section is written using *only* retrieved evidence passages |
| **Hallucination control** | Unsupported claims are suppressed — model writes `"No clear evidence found"` instead of fabricating |
| **Retrieval provenance** | Every claim carries Evidence objects citing `[doc_id, p.N]` |
| **Operator edit loop** | Diffs original vs. edited draft; extracts reusable patterns; applies them to future drafts |
| **Frequency gating** | Patterns are only applied after `MIN_EDITS_TO_LEARN` occurrences — prevents overfitting to a single edit |
| **REST API** | FastAPI endpoints for upload, generate, retrieve, and feedback |

---

## Hallucination Control — Highlighted

This system enforces a **retrieval-constrained generation** policy:

1. **Evidence mandatory** — Llama 3 is given *only* retrieved passages as source material.  
2. **Unsupported claims suppressed** — The system prompt instructs the model to write `"No clear evidence found"` for any section where retrieval returns nothing relevant.  
3. **Inspectable at claim level** — Every sentence in the generated memo carries the `Evidence` objects that supported it, enabling reviewers to verify each statement against the source document.

This is not a typical RAG pipeline that retrieves context and hopes the model stays grounded. The prompt engineering *enforces* groundedness as a hard rule.

---

## Modules

| File | Internal name | Role |
|---|---|---|
| `processor.py` | processor | Document ingestion & OCR |
| `retriever.py` | retriever | Embeddings & FAISS index |
| `generator.py` | generator | Grounded memo generation |
| `feedback.py` | feedback | Operator edit learning |
| `config.py` | config | Central configuration |
| `pipeline.py` | **orchestrator** | End-to-end entry point (CLI) |
| `api.py` | api | FastAPI REST interface |

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Tesseract** must be installed separately for OCR on scanned PDFs:
> - Windows: https://github.com/UB-Mannheim/tesseract/wiki
> - Linux: `sudo apt install tesseract-ocr`

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

### 3. Generate sample documents

```bash
python create_sample_pdf.py
```

This creates `sample_docs/sample_contract.pdf` and all example outputs in `sample_outputs/`.

### 4. Generate the architecture diagram

```bash
python create_diagram.py
```

### 5. Run the full pipeline demo

```bash
python pipeline.py --demo
```

### 6. Run on your own PDF

```bash
# Ingest + generate memo
python pipeline.py --pdf path/to/contract.pdf

# Query the evidence index
python pipeline.py --doc-id contract --query "termination conditions"

# Record an operator edit (triggers pattern learning)
python pipeline.py --doc-id contract --record-edit \
    --original original_draft.txt --edited edited_draft.txt
```

### 7. Start the API server

```bash
uvicorn api:app --reload --port 8000
```

API docs: http://localhost:8000/docs

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/upload` | Upload & ingest a PDF |
| `POST` | `/generate` | Generate grounded memo |
| `POST` | `/retrieve` | Semantic evidence retrieval |
| `POST` | `/feedback` | Record operator edit |
| `GET` | `/patterns` | List learned style patterns |
| `GET` | `/docs` | List ingested documents |
| `GET` | `/health` | System health + index stats |

---

## Sample Inputs & Outputs

Pre-generated examples (no API key required to inspect):

| File | Contents |
|---|---|
| [`sample_docs/sample_contract.pdf`](sample_docs/sample_contract.pdf) | Synthetic NDA / service agreement |
| [`sample_outputs/processed_doc.json`](sample_outputs/processed_doc.json) | Ingestion output: pages + chunks |
| [`sample_outputs/retrieval_example.json`](sample_outputs/retrieval_example.json) | Evidence retrieval for a sample query |
| [`sample_outputs/memo_output.json`](sample_outputs/memo_output.json) | Full grounded memo with evidence citations |
| [`sample_outputs/learned_patterns.json`](sample_outputs/learned_patterns.json) | Patterns extracted from operator edits |

---

## Design Tradeoffs

### Claim-evidence matching is heuristic
The `_parse_grounded_claims` function uses word-overlap to link generated sentences to evidence chunks. This is a deliberate simplification — production systems would use semantic similarity here. This heuristic is fast, dependency-free, and sufficient for demonstrating the architecture. The LLM is already instructed to cite sources inline via `[doc_id, p.N]`.

### OCR fallback is partially limited
When a PDF page contains no extractable text and no embedded images, the code creates a placeholder white image before OCR. This handles the common case of malformed PDFs but will not rasterise vector-rendered pages. For production deployments, replace with `pdf2image` or `PyMuPDF` (`fitz`) for true page rasterisation:

```python
# Production rasterisation (replace placeholder in processor.py)
import fitz
doc = fitz.open(pdf_path)
pix = doc[page_index].get_pixmap(dpi=300)
img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
```

### Pattern learning requires multiple edits
Patterns are applied only after `MIN_EDITS_TO_LEARN = 2` occurrences. This prevents a single idiosyncratic edit from contaminating future drafts. In production, this threshold should be tunable per organisation.

---

## Configuration

All tuneable constants are in `config.py`. Override via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Model to use for generation |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `CHUNK_SIZE` | `400` | Tokens per chunk |
| `CHUNK_OVERLAP` | `80` | Overlap tokens between chunks |
| `TOP_K` | `6` | Evidence passages per memo section |
| `MIN_SCORE` | `0.25` | Cosine similarity floor |
| `MIN_EDITS_TO_LEARN` | `2` | Min occurrences before pattern applies |

---

## Project Structure

```
.
├── pipeline.py           ← Orchestrator / entry point
├── processor.py          ← Document ingestion & OCR
├── retriever.py          ← Embeddings & FAISS retrieval
├── generator.py          ← Grounded memo generation
├── feedback.py           ← Operator edit learning loop
├── config.py             ← Central configuration
├── api.py                ← FastAPI REST interface
├── create_sample_pdf.py  ← Sample document generator
├── create_diagram.py     ← Architecture diagram generator
├── requirements.txt
├── .env.example
├── architecture.txt      ← ASCII architecture diagram
├── architecture.png      ← PNG diagram (after running create_diagram.py)
├── sample_docs/
│   ├── sample_contract.pdf
│   ├── sample_original.txt
│   └── sample_edited.txt
├── sample_outputs/
│   ├── processed_doc.json
│   ├── retrieval_example.json
│   ├── memo_output.json
│   └── learned_patterns.json
└── data/
    ├── uploads/
    ├── processed/         ← Ingested document JSON cache
    ├── vector_store/      ← FAISS index + metadata sidecar
    └── feedback/          ← Edit records + learned patterns
```
