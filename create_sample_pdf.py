# -*- coding: utf-8 -*-
"""
create_sample_pdf.py — Generates a realistic sample PDF and example outputs

Run once to bootstrap the demo:
    python create_sample_pdf.py

Creates:
    sample_docs/sample_contract.pdf   — synthetic NDA/service agreement
    sample_docs/sample_original.txt   — an AI-drafted memo section
    sample_docs/sample_edited.txt     — the same section after operator edits
    sample_outputs/processed_doc.json — what the ingestion step produces
    sample_outputs/retrieval_example.json  — sample retrieval results
    sample_outputs/memo_output.json        — sample grounded memo
    sample_outputs/learned_patterns.json   — sample learned patterns
"""

from __future__ import annotations

import json
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parent
SAMPLE = BASE / "sample_docs"
OUT    = BASE / "sample_outputs"
SAMPLE.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
#  1.  Sample PDF (synthetic NDA / service agreement)
# ══════════════════════════════════════════════════════════════════════════

PDF_TEXT = """\
CONFIDENTIAL — INTERNAL MEMORANDUM

MASTER SERVICES AND NON-DISCLOSURE AGREEMENT
Between:  Apex Holdings LLC ("Client")
And:      Meridian Consulting Group Inc. ("Consultant")
Date:     15 January 2025
Contract Ref: MSA-2025-007

──────────────────────────────────────────────────────────────
1. SCOPE OF SERVICES

1.1  Consultant shall provide strategic advisory services in connection with
the Client's proposed acquisition of Drayfield Technologies Ltd, including:
     (a) Due-diligence review of financial statements (FY2022–FY2024);
     (b) Regulatory risk assessment under the Competition Act 2010;
     (c) Integration planning and post-merger operational recommendations.

1.2  Services shall commence on 1 February 2025 and terminate no later
than 31 July 2025 unless extended by written agreement.

──────────────────────────────────────────────────────────────
2. FEES AND PAYMENT

2.1  Client shall pay a fixed monthly retainer of USD 45,000, invoiced on
the first business day of each month.

2.2  Out-of-pocket expenses (travel, accommodation) are reimbursable upon
submission of receipts; individual expenses exceeding USD 2,500 require
prior written approval from Client's CFO.

2.3  Invoices unpaid after 30 days accrue interest at 1.5 % per month.
     Client's obligation to pay fees is unconditional and not subject to
     any right of set-off.

──────────────────────────────────────────────────────────────
3. CONFIDENTIALITY

3.1  "Confidential Information" means all non-public information disclosed
by either party relating to Drayfield Technologies Ltd., the proposed
transaction, or the Client's business operations.

3.2  Consultant undertakes to: (a) hold Confidential Information in strict
confidence; (b) use it solely for the Purpose; (c) not disclose it to any
third party without prior written consent.

3.3  These obligations survive termination for a period of five (5) years.

──────────────────────────────────────────────────────────────
4. INTELLECTUAL PROPERTY

4.1  All deliverables, reports, and work product created by Consultant under
this Agreement shall be the sole and exclusive property of Client upon
full payment of applicable fees.

4.2  Consultant retains ownership of its pre-existing proprietary frameworks
and methodologies; Client receives a perpetual, royalty-free licence to
use them solely in connection with deliverables under this Agreement.

──────────────────────────────────────────────────────────────
5. LIABILITY AND INDEMNIFICATION

5.1  Consultant's aggregate liability under this Agreement shall not exceed
the total fees paid in the three (3) months immediately preceding the claim.

5.2  Neither party shall be liable for indirect, consequential, or punitive
damages arising from this Agreement.

5.3  Client shall indemnify and hold harmless Consultant from any third-party
claims arising from Client's misuse of the deliverables.

──────────────────────────────────────────────────────────────
6. TERMINATION

6.1  Either party may terminate this Agreement on 30 days' written notice.

6.2  Client may terminate immediately for Consultant's material breach, if
the breach is not cured within 15 days of written notice.

6.3  On termination: (a) all outstanding fees become immediately due;
(b) each party shall return or destroy Confidential Information.

──────────────────────────────────────────────────────────────
7. GOVERNING LAW

This Agreement shall be governed by and construed in accordance with the
laws of the State of New York. Disputes shall be resolved by binding
arbitration in New York City under the JAMS rules.

──────────────────────────────────────────────────────────────
SIGNATURES

Apex Holdings LLC                  Meridian Consulting Group Inc.
____________________________       ____________________________
James R. Whitmore (CEO)            Sandra L. Park (Managing Director)
Date: 15 January 2025              Date: 15 January 2025
"""


def create_sample_pdf():
    """Create a sample PDF using reportlab (preferred) or fpdf2 fallback."""
    pdf_path = SAMPLE / "sample_contract.pdf"

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.enums import TA_LEFT

        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        body   = styles["BodyText"]
        body.fontSize  = 10
        body.leading   = 14
        body.fontName  = "Helvetica"

        story = []
        for line in PDF_TEXT.split("\n"):
            stripped = line.strip()
            if not stripped:
                story.append(Spacer(1, 6))
            elif stripped.startswith("──"):
                story.append(Spacer(1, 4))
            else:
                story.append(Paragraph(stripped.replace("&", "&amp;"), body))
        doc.build(story)
        print(f"✓  Created sample PDF (reportlab): {pdf_path}")

    except ImportError:
        # Fallback: fpdf2
        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.set_margins(20, 20, 20)
            pdf.add_page()
            pdf.set_font("Helvetica", size=10)
            for line in PDF_TEXT.split("\n"):
                if line.strip().startswith("──"):
                    pdf.ln(3)
                else:
                    pdf.multi_cell(0, 6, line)
            pdf.output(str(pdf_path))
            print(f"✓  Created sample PDF (fpdf2): {pdf_path}")

        except ImportError:
            # Last resort: write raw minimal PDF manually
            _write_minimal_pdf(pdf_path)

    return pdf_path


def _write_minimal_pdf(path: Path):
    """Write a bare-minimum valid PDF containing the contract text."""
    lines = PDF_TEXT.replace("(", r"\(").replace(")", r"\)").split("\n")
    pdf_lines = []
    y = 750
    content_stream = "BT\n/F1 9 Tf\n"
    for line in lines:
        content_stream += f"72 {y} Td ({line[:100]}) Tj\n"
        y -= 13
        if y < 60:
            y = 750
    content_stream += "ET\n"

    body  = content_stream.encode("latin-1", errors="replace")
    xref1 = len(b"%PDF-1.4\n")
    raw = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n"
        b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n"
        b"3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>>>>\nendobj\n"
        + f"4 0 obj\n<</Length {len(body)}>>\nstream\n".encode()
        + body + b"\nendstream\nendobj\n"
        b"5 0 obj\n<</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n"
        b"trailer\n<</Size 6 /Root 1 0 R>>\nstartxref\n9\n%%EOF\n"
    )
    path.write_bytes(raw)
    print(f"✓  Created sample PDF (minimal): {path}")


# ══════════════════════════════════════════════════════════════════════════
#  2.  Sample operator edit files (for feedback demo)
# ══════════════════════════════════════════════════════════════════════════

ORIGINAL_DRAFT = """\
The contract specifies a fixed monthly payment of USD 45,000.
Unpaid invoices after 30 days will have interest added at 1.5% per month.
The agreement can be ended by either party with 30 days notice.
The liability of the consultant is capped at recent fees paid.
Confidentiality rules remain in force for 5 years after the agreement ends.
"""

EDITED_DRAFT = """\
The Agreement stipulates a fixed monthly retainer of USD 45,000.
Invoices remaining unpaid after 30 days shall accrue interest at the rate of 1.5% per month.
Either party may terminate the Agreement upon 30 days' written notice.
Consultant's aggregate liability is limited to fees paid in the preceding three months.
Confidentiality obligations survive termination for a period of five (5) years.
"""


# ══════════════════════════════════════════════════════════════════════════
#  3.  Static sample outputs (reviewers can inspect without running pipeline)
# ══════════════════════════════════════════════════════════════════════════

PROCESSED_DOC_SAMPLE = {
    "doc_id": "sample_contract",
    "filename": "sample_contract.pdf",
    "file_type": "PDF",
    "total_pages": 2,
    "processed_at": "2025-05-15T09:12:34Z",
    "extraction_summary": {
        "native_pages": 2,
        "ocr_pages": 0,
        "total_chunks": 8,
        "elapsed_s": 0.43
    },
    "pages": [
        {
            "page_number": 1,
            "extraction_method": "native",
            "char_count": 1842,
            "confidence": None,
            "text": "MASTER SERVICES AND NON-DISCLOSURE AGREEMENT\nBetween: Apex Holdings LLC (\"Client\") and Meridian Consulting Group Inc. (\"Consultant\")\n..."
        }
    ],
    "chunks": [
        {
            "chunk_id": "sample_contract_c0",
            "doc_id": "sample_contract",
            "text": "MASTER SERVICES AND NON-DISCLOSURE AGREEMENT\nBetween: Apex Holdings LLC...\nScope: strategic advisory for proposed acquisition of Drayfield Technologies Ltd.",
            "start_page": 1,
            "end_page": 1,
            "chunk_index": 0
        },
        {
            "chunk_id": "sample_contract_c1",
            "doc_id": "sample_contract",
            "text": "2. FEES AND PAYMENT\nClient shall pay a fixed monthly retainer of USD 45,000...\nInvoices unpaid after 30 days accrue interest at 1.5% per month.",
            "start_page": 1,
            "end_page": 1,
            "chunk_index": 1
        }
    ]
}

RETRIEVAL_EXAMPLE = {
    "query": "payment obligations and interest on late invoices",
    "results": [
        {
            "chunk_id": "sample_contract_c1",
            "doc_id": "sample_contract",
            "score": 0.8731,
            "start_page": 1,
            "end_page": 1,
            "text": "2. FEES AND PAYMENT — Client shall pay a fixed monthly retainer of USD 45,000, invoiced on the first business day of each month. Invoices unpaid after 30 days accrue interest at 1.5% per month...",
            "citation": "[sample_contract, p.1]"
        },
        {
            "chunk_id": "sample_contract_c4",
            "doc_id": "sample_contract",
            "score": 0.6218,
            "start_page": 2,
            "end_page": 2,
            "text": "6. TERMINATION — On termination: (a) all outstanding fees become immediately due...",
            "citation": "[sample_contract, p.2]"
        }
    ]
}

MEMO_OUTPUT_SAMPLE = {
    "doc_ids": ["sample_contract"],
    "model": "claude-sonnet-4-5",
    "version": 1,
    "patterns_applied": [],
    "sections": [
        {
            "title": "Summary",
            "raw_text": (
                "This memorandum analyses the Master Services and Non-Disclosure Agreement (MSA-2025-007) "
                "between Apex Holdings LLC (Client) and Meridian Consulting Group Inc. (Consultant), "
                "dated 15 January 2025. The Agreement governs advisory services for Client's proposed "
                "acquisition of Drayfield Technologies Ltd., with services running from February to "
                "July 2025. [sample_contract, p.1]"
            ),
            "claims": [
                {
                    "statement": "This memorandum analyses the Master Services and Non-Disclosure Agreement (MSA-2025-007).",
                    "evidence": [{"chunk_id": "sample_contract_c0", "doc_id": "sample_contract",
                                  "score": 0.912, "start_page": 1, "end_page": 1,
                                  "text": "MASTER SERVICES AND NON-DISCLOSURE AGREEMENT..."}]
                }
            ]
        },
        {
            "title": "Key Facts",
            "raw_text": (
                "• Contract Ref: MSA-2025-007, effective 15 January 2025. [sample_contract, p.1]\n"
                "• Fixed monthly retainer: USD 45,000. [sample_contract, p.1]\n"
                "• Service period: 1 February 2025 – 31 July 2025. [sample_contract, p.1]\n"
                "• Liability cap: fees paid in the preceding 3 months. [sample_contract, p.2]\n"
                "• Confidentiality obligations survive for 5 years post-termination. [sample_contract, p.1]"
            ),
            "claims": []
        },
        {
            "title": "Potential Risks",
            "raw_text": (
                "1. Liability exposure is limited to three months of fees, which may be insufficient "
                "given the scale of the proposed acquisition. [sample_contract, p.2]\n"
                "2. The 30-day termination notice is short for a complex M&A advisory engagement. [sample_contract, p.2]\n"
                "3. No clear evidence found regarding data breach liability or cybersecurity obligations."
            ),
            "claims": []
        },
        {
            "title": "Missing Information",
            "raw_text": (
                "• No clear evidence found of a dispute resolution timeline for arbitration proceedings.\n"
                "• The Agreement lacks explicit data protection / GDPR provisions.\n"
                "• Success fee or completion bonus mechanism not addressed."
            ),
            "claims": []
        }
    ]
}

LEARNED_PATTERNS_SAMPLE = [
    {
        "pattern_id": "pat_0001",
        "type": "terminology",
        "description": "Prefer 'retainer' over 'payment' for fixed monthly fees",
        "from_text": "payment",
        "to_text": "retainer",
        "frequency": 3,
        "examples": ["edit_0001", "edit_0002", "edit_0003"]
    },
    {
        "pattern_id": "pat_0002",
        "type": "tone",
        "description": "Use formal legal phrasing: 'shall' instead of 'will'",
        "from_text": "will",
        "to_text": "shall",
        "frequency": 5,
        "examples": ["edit_0001", "edit_0002"]
    },
    {
        "pattern_id": "pat_0003",
        "type": "structure",
        "description": "Cite specific clause numbers when referencing contract provisions",
        "from_text": "the agreement states",
        "to_text": "Clause N.N provides that",
        "frequency": 2,
        "examples": ["edit_0003"]
    }
]


def write_sample_outputs():
    files = {
        OUT / "processed_doc.json":       PROCESSED_DOC_SAMPLE,
        OUT / "retrieval_example.json":   RETRIEVAL_EXAMPLE,
        OUT / "memo_output.json":         MEMO_OUTPUT_SAMPLE,
        OUT / "learned_patterns.json":    LEARNED_PATTERNS_SAMPLE,
    }
    for path, data in files.items():
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"✓  Written: {path.relative_to(BASE)}")


if __name__ == "__main__":
    print("Creating sample documents and example outputs …\n")
    create_sample_pdf()

    (SAMPLE / "sample_original.txt").write_text(ORIGINAL_DRAFT, encoding="utf-8")
    (SAMPLE / "sample_edited.txt").write_text(EDITED_DRAFT, encoding="utf-8")
    print(f"✓  Written: sample_docs/sample_original.txt")
    print(f"✓  Written: sample_docs/sample_edited.txt")

    write_sample_outputs()
    print("\nDone.  Run:  python pipeline.py --demo")
