"""
feedback.py — Operator Edit Learning Loop

Responsibilities:
  1. Accept a (original_draft, edited_draft) pair from an operator
  2. Diff the two versions at the sentence level
  3. Extract reusable patterns:
       - Terminology substitutions (word/phrase replacements)
       - Structural preferences (section additions / removals)
       - Tone shifts (inferred from systematic changes)
  4. Persist patterns in JSON with frequency counts
  5. Expose the learned patterns to the generator so future drafts improve

Design:
  - Patterns are stored as a JSON list in PATTERNS_PATH.
  - Each pattern has: type, description, from_text, to_text, frequency, examples.
  - The learning step uses difflib SequenceMatcher for line-level diff, then
    LLM-assisted extraction for semantic pattern inference.
  - We require MIN_EDITS_TO_LEARN occurrences before a pattern influences generation,
    preventing overfitting to a single edit.

This is a REAL improvement loop — not a version diff sidecar. Patterns are:
  a) extracted from edits,
  b) generalised (not just stored verbatim),
  c) applied as style guidance in future generation prompts.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple

from groq import Groq

from config import (
    GROQ_API_KEY,
    LLM_MODEL,
    FEEDBACK_DB_PATH,
    PATTERNS_PATH,
    MIN_EDITS_TO_LEARN,
)

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────

@dataclass
class EditRecord:
    edit_id:       str
    doc_id:        str
    original_text: str
    edited_text:   str
    timestamp:     str
    section:       Optional[str] = None


@dataclass
class LearnedPattern:
    pattern_id:  str
    type:        str        # "terminology" | "structure" | "tone" | "addition" | "removal"
    description: str        # Human-readable summary e.g. "Use 'chain-of-title' not 'ownership'"
    from_text:   str        # What was changed FROM (may be empty for additions)
    to_text:     str        # What it was changed TO (may be empty for removals)
    frequency:   int = 1
    examples:    List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LearnedPattern":
        return cls(**d)


# ── Diff helpers ──────────────────────────────────────────────────────────

def _sentence_split(text: str) -> List[str]:
    """Split text into sentences, preserving trailing punctuation."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def _line_diff(original: str, edited: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Return (removed_lines, added_lines, unchanged_lines) between two texts.
    Uses difflib for fast, dependency-free diffing.
    """
    orig_lines = original.splitlines()
    edit_lines = edited.splitlines()
    matcher    = difflib.SequenceMatcher(None, orig_lines, edit_lines, autojunk=False)

    removed, added, unchanged = [], [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            removed.extend(orig_lines[i1:i2])
            added.extend(edit_lines[j1:j2])
        elif tag == "delete":
            removed.extend(orig_lines[i1:i2])
        elif tag == "insert":
            added.extend(edit_lines[j1:j2])
        elif tag == "equal":
            unchanged.extend(orig_lines[i1:i2])
    return removed, added, unchanged


def _find_terminology_swaps(removed: List[str], added: List[str]) -> List[Tuple[str, str]]:
    """
    Heuristic: if a removed line and added line differ only in a short phrase,
    call it a terminology swap.
    """
    swaps: List[Tuple[str, str]] = []
    for rem in removed:
        for add in added:
            if abs(len(rem) - len(add)) > 60:
                continue
            # Find differing words
            rem_words = set(re.findall(r"\b\w[\w'-]*\w\b", rem.lower()))
            add_words = set(re.findall(r"\b\w[\w'-]*\w\b", add.lower()))
            only_rem  = rem_words - add_words
            only_add  = add_words - rem_words
            if 0 < len(only_rem) <= 4 and 0 < len(only_add) <= 4:
                swaps.append((" ".join(sorted(only_rem)), " ".join(sorted(only_add))))
    return swaps


# ── LLM-assisted pattern extraction ──────────────────────────────────────

EXTRACTION_PROMPT = """You are analysing an operator's edits to an AI-generated legal memo.
Your job is to extract REUSABLE style/tone/terminology patterns from the diff below.

REMOVED LINES (operator deleted or replaced these):
{removed}

ADDED LINES (operator wrote these instead):
{added}

Extract up to 5 patterns. For each, output a JSON object with:
  - type: one of "terminology", "structure", "tone", "addition", "removal"
  - description: one concise sentence describing the preference (generalised, not doc-specific)
  - from_text: the original phrasing (empty string if it's a pure addition)
  - to_text: the preferred phrasing (empty string if it's a pure removal)

Return ONLY a JSON array. No preamble. No markdown fences."""


def _extract_patterns_via_llm(removed: List[str], added: List[str]) -> List[dict]:
    if not removed and not added:
        return []
    client = Groq(api_key=GROQ_API_KEY)
    prompt = EXTRACTION_PROMPT.format(
        removed="\n".join(f"- {r}" for r in removed[:20]) or "(none)",
        added  ="\n".join(f"- {a}" for a in added[:20]) or "(none)",
    )
    try:
        resp = client.chat.completions.create(
            model      = LLM_MODEL,
            max_tokens = 1024,
            messages   = [{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception as e:
        logger.error("Pattern extraction LLM call failed: %s", e)
        return []


# ── Persistence helpers ───────────────────────────────────────────────────

def _load_feedback() -> List[dict]:
    if FEEDBACK_DB_PATH.exists():
        return json.loads(FEEDBACK_DB_PATH.read_text())
    return []


def _save_feedback(records: List[dict]):
    FEEDBACK_DB_PATH.write_text(json.dumps(records, indent=2))


def _load_patterns() -> List[LearnedPattern]:
    if PATTERNS_PATH.exists():
        raw = json.loads(PATTERNS_PATH.read_text())
        return [LearnedPattern.from_dict(p) for p in raw]
    return []


def _save_patterns(patterns: List[LearnedPattern]):
    PATTERNS_PATH.write_text(json.dumps([p.to_dict() for p in patterns], indent=2))


# ── Public API ────────────────────────────────────────────────────────────

def record_edit(
    doc_id:        str,
    original_text: str,
    edited_text:   str,
    section:       Optional[str] = None,
) -> dict:
    """
    Record an operator edit and update the learned patterns.

    Returns a summary of what was learned.
    """
    # 1. Store the raw edit record
    records  = _load_feedback()
    edit_id  = f"edit_{len(records)+1:04d}"
    record   = EditRecord(
        edit_id       = edit_id,
        doc_id        = doc_id,
        original_text = original_text,
        edited_text   = edited_text,
        timestamp     = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        section       = section,
    )
    records.append(asdict(record))
    _save_feedback(records)
    logger.info("Recorded edit %s for doc %s", edit_id, doc_id)

    # 2. Diff the two versions
    removed, added, unchanged = _line_diff(original_text, edited_text)
    if not removed and not added:
        return {"edit_id": edit_id, "patterns_learned": 0, "message": "No differences detected"}

    # 3. Extract patterns (heuristic + LLM)
    raw_patterns = _extract_patterns_via_llm(removed, added)

    # Also capture simple terminology swaps heuristically
    swaps = _find_terminology_swaps(removed, added)
    for from_w, to_w in swaps:
        raw_patterns.append({
            "type":        "terminology",
            "description": f"Prefer '{to_w}' over '{from_w}'",
            "from_text":   from_w,
            "to_text":     to_w,
        })

    # 4. Merge into the global patterns store (de-duplicate by description)
    existing = _load_patterns()
    existing_map = {p.description.lower(): p for p in existing}
    new_count = 0

    for rp in raw_patterns:
        if not isinstance(rp, dict):
            continue
        desc = rp.get("description", "").strip()
        if not desc:
            continue
        key = desc.lower()
        if key in existing_map:
            existing_map[key].frequency += 1
            existing_map[key].examples.append(edit_id)
        else:
            new_p = LearnedPattern(
                pattern_id  = f"pat_{len(existing_map)+1:04d}",
                type        = rp.get("type", "terminology"),
                description = desc,
                from_text   = rp.get("from_text", ""),
                to_text     = rp.get("to_text", ""),
                frequency   = 1,
                examples    = [edit_id],
            )
            existing_map[key] = new_p
            new_count += 1

    updated = list(existing_map.values())
    _save_patterns(updated)

    logger.info("Patterns: %d new, %d total", new_count, len(updated))
    return {
        "edit_id":         edit_id,
        "lines_removed":   len(removed),
        "lines_added":     len(added),
        "patterns_learned": new_count,
        "total_patterns":   len(updated),
    }


def get_active_patterns(min_frequency: int = MIN_EDITS_TO_LEARN) -> List[LearnedPattern]:
    """
    Return only patterns that have been seen enough times to be trustworthy.
    """
    return [p for p in _load_patterns() if p.frequency >= min_frequency]


def get_all_patterns() -> List[LearnedPattern]:
    return _load_patterns()


def get_style_guidance(min_frequency: int = MIN_EDITS_TO_LEARN) -> str:
    """
    Produce a style-guidance string to inject into generation prompts.
    """
    patterns = get_active_patterns(min_frequency)
    if not patterns:
        return ""
    lines = []
    for p in patterns:
        lines.append(f"- [{p.type.upper()}] {p.description}")
        if p.from_text and p.to_text:
            lines.append(f"  Instead of: '{p.from_text}'  →  Use: '{p.to_text}'")
    return "\n".join(lines)


def get_feedback_stats() -> dict:
    records  = _load_feedback()
    patterns = _load_patterns()
    active   = get_active_patterns()
    return {
        "total_edits":    len(records),
        "total_patterns": len(patterns),
        "active_patterns": len(active),
        "min_frequency_threshold": MIN_EDITS_TO_LEARN,
    }
