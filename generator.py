"""
generator.py — Grounded Legal Memo Generator

Responsibilities:
  1. Accept a drafting task (doc IDs + optional context)
  2. Retrieve targeted evidence per memo section
  3. Call the Claude API with section-specific prompts + injected evidence
  4. Return a fully grounded MemoOutput with per-claim evidence citations
  5. Apply any learned style patterns from the feedback loop

The memo structure follows the Internal Legal Memo format recommended in the
approach document:
  Summary | Key Facts | Potential Risks | Missing Information | Supporting Evidence

Every claim in the output is paired with the Evidence objects that support it.
This makes the output inspectable at the chunk level — reviewers can verify
exactly which passage supported which statement.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from groq import Groq

from config import GROQ_API_KEY, LLM_MODEL, MAX_TOKENS, TOP_K
from retriever import Evidence, Retriever

logger = logging.getLogger(__name__)


# ── Output models ─────────────────────────────────────────────────────────

@dataclass
class GroundedClaim:
    """A single statement in the memo, paired with supporting evidence."""
    statement:  str
    evidence:   List[Evidence] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "statement": self.statement,
            "evidence":  [e.to_dict() for e in self.evidence],
        }


@dataclass
class MemoSection:
    title:  str
    claims: List[GroundedClaim] = field(default_factory=list)
    raw_text: str = ""          # full LLM output for this section

    def to_dict(self) -> dict:
        return {
            "title":    self.title,
            "raw_text": self.raw_text,
            "claims":   [c.to_dict() for c in self.claims],
        }


@dataclass
class MemoOutput:
    doc_ids:   List[str]
    sections:  List[MemoSection]
    model:     str
    version:   int = 1          # bumped each time operator edits are applied
    patterns_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "doc_ids":          self.doc_ids,
            "model":            self.model,
            "version":          self.version,
            "patterns_applied": self.patterns_applied,
            "sections":         [s.to_dict() for s in self.sections],
        }

    def as_plain_text(self) -> str:
        lines = []
        for s in self.sections:
            lines.append(f"\n## {s['title']}" if isinstance(s, dict) else f"\n## {s.title}")
            text = s["raw_text"] if isinstance(s, dict) else s.raw_text
            lines.append(text)
        return "\n".join(lines)


# ── Prompt templates ──────────────────────────────────────────────────────

SECTION_QUERIES = {
    "Summary":              "executive summary overview main findings",
    "Key Facts":            "key facts parties dates amounts obligations conditions",
    "Potential Risks":      "risks liabilities disputes issues problems concerns violations",
    "Missing Information":  "missing information gaps unclear ambiguous incomplete",
    "Supporting Evidence":  "evidence supporting material documents exhibits references",
}

SYSTEM_PROMPT = """You are a senior legal analyst at Pearson Specter Litt producing an internal legal memo.
Your task is to write a specific section of the memo based ONLY on the provided source evidence.

Rules:
1. Ground every statement in the provided evidence passages. Do not invent facts.
2. If the evidence does not support a claim, state "No clear evidence found" rather than speculating.
3. Be precise and professional. Use proper legal terminology.
4. Reference source documents inline using the format [doc_id, p.N].
5. Keep your response focused on this section only.
{style_guidance}"""

SECTION_PROMPT = """Write the "{section}" section of an internal legal memo.

=== EVIDENCE PASSAGES ===
{evidence_block}
========================

Instructions:
- Use ONLY the evidence above as your source material.
- For each material claim, note the source reference in brackets.
- If evidence is limited or contradictory, flag it explicitly.
- Respond with the section content only (no heading needed).
"""


def _build_evidence_block(evidence_list: List[Evidence]) -> str:
    if not evidence_list:
        return "[No relevant passages retrieved]"
    lines = []
    for i, e in enumerate(evidence_list, 1):
        lines.append(
            f"[{i}] {e.citation()}\n"
            f"Score: {e.score:.2f}\n"
            f"{e.text[:600]}{'...' if len(e.text) > 600 else ''}\n"
        )
    return "\n---\n".join(lines)


def _parse_grounded_claims(text: str, evidence_list: List[Evidence]) -> List[GroundedClaim]:
    """
    Heuristic: split the generated text into sentences and attach the
    evidence that was retrieved for this section (all evidence is relevant
    context; the LLM was instructed to use it).
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    claims = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Find evidence whose text overlaps meaningfully with the claim
        matched = [e for e in evidence_list if any(
            word.lower() in e.text.lower()
            for word in sent.split()
            if len(word) > 5
        )]
        claims.append(GroundedClaim(statement=sent, evidence=matched[:3]))
    return claims


# ── Generator ─────────────────────────────────────────────────────────────

class MemoGenerator:
    def __init__(self, retriever: Retriever):
        self._retriever = retriever
        self._client    = Groq(api_key=GROQ_API_KEY)

    def _call_llm(self, system: str, user: str) -> str:
        try:
            resp = self._client.chat.completions.create(
                model      = LLM_MODEL,
                max_tokens = MAX_TOKENS,
                messages   = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error("LLM API error: %s", e)
            raise

    def generate(
        self,
        doc_ids:          List[str],
        style_guidance:   str = "",
        top_k:            int = TOP_K,
    ) -> MemoOutput:
        """
        Generate a full grounded internal legal memo for the given documents.

        Args:
            doc_ids:        Documents to base the memo on
            style_guidance: Injected style preferences from the learning loop
            top_k:          Passages to retrieve per section

        Returns:
            MemoOutput with all sections and grounded evidence.
        """
        logger.info("Generating memo for doc_ids=%s", doc_ids)

        system = SYSTEM_PROMPT.format(
            style_guidance=f"\nStyle guidance from previous edits:\n{style_guidance}" if style_guidance else ""
        )

        sections: List[MemoSection] = []

        for section_title, query in SECTION_QUERIES.items():
            evidence = self._retriever.retrieve(
                query   = query,
                top_k   = top_k,
                doc_ids = doc_ids,
            )
            evidence_block = _build_evidence_block(evidence)
            user_msg = SECTION_PROMPT.format(
                section        = section_title,
                evidence_block = evidence_block,
            )
            logger.info("Generating section: %s (evidence count: %d)", section_title, len(evidence))
            raw_text = self._call_llm(system, user_msg)
            claims   = _parse_grounded_claims(raw_text, evidence)

            sections.append(MemoSection(
                title    = section_title,
                claims   = claims,
                raw_text = raw_text,
            ))

        return MemoOutput(
            doc_ids  = doc_ids,
            sections = sections,
            model    = LLM_MODEL,
        )


# ── Quick regeneration with learned patterns ──────────────────────────────

def apply_learned_patterns(patterns: List[dict]) -> str:
    """
    Convert learned patterns into a concise style-guidance block
    to inject into the system prompt.
    """
    if not patterns:
        return ""
    lines = ["Apply these style preferences based on prior operator edits:"]
    for p in patterns:
        lines.append(f"- {p['description']} (seen {p['frequency']}x)")
    return "\n".join(lines)
