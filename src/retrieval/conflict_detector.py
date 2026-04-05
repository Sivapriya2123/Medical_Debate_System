"""Conflict detection — flag contradictory claims in retrieved evidence.

Uses two complementary signals:
  1. Semantic similarity: high cosine similarity between evidence snippets
     suggests they discuss the same topic.
  2. Negation patterns: if similar snippets contain opposing language
     (e.g. "effective" vs "not effective"), they likely conflict.

The combination avoids false positives from unrelated evidence and
false negatives from subtle contradictions.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# ── Negation / Contradiction Patterns ────────────────────────────

_NEGATION_PAIRS = [
    (r"\beffective\b", r"\bnot effective\b"),
    (r"\beffective\b", r"\bineffective\b"),
    (r"\bsignificant\b", r"\bnot significant\b"),
    (r"\bsignificant\b", r"\binsignificant\b"),
    (r"\bincreased?\b", r"\bdecreased?\b"),
    (r"\bhigher\b", r"\blower\b"),
    (r"\bimproved?\b", r"\bworsened?\b"),
    (r"\bbeneficial\b", r"\bharmful\b"),
    (r"\bpositive\b", r"\bnegative\b"),
    (r"\bsafe\b", r"\bunsafe\b"),
    (r"\brecommended\b", r"\bnot recommended\b"),
    (r"\bsupports?\b", r"\bcontradicts?\b"),
    (r"\bassociated\b", r"\bnot associated\b"),
    (r"\bcorrelated\b", r"\bnot correlated\b"),
    (r"\bconfirm(?:s|ed)?\b", r"\bdeny|denied|denies\b"),
    (r"\byes\b", r"\bno\b"),
]


def _has_negation_conflict(text_a: str, text_b: str) -> bool:
    """Check if two texts contain opposing language patterns."""
    a_lower = text_a.lower()
    b_lower = text_b.lower()

    for pat_pos, pat_neg in _NEGATION_PAIRS:
        # A has positive, B has negative
        if re.search(pat_pos, a_lower) and re.search(pat_neg, b_lower):
            return True
        # A has negative, B has positive
        if re.search(pat_neg, a_lower) and re.search(pat_pos, b_lower):
            return True

    return False


# ── Core Conflict Detection ──────────────────────────────────────

_embedding_model: Optional[SentenceTransformer] = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def detect_conflicts(
    evidence_list: List[Dict[str, Any]],
    similarity_threshold: float = 0.7,
    text_key: str = "text",
) -> List[Dict[str, Any]]:
    """Detect conflicting pairs in a list of evidence items.

    Two evidence items conflict when:
      - Their embeddings have cosine similarity >= similarity_threshold
        (i.e. they discuss the same topic)
      - AND they contain opposing language patterns

    Args:
        evidence_list: List of evidence dicts (each must have a text field).
        similarity_threshold: Min cosine sim to consider same-topic.
        text_key: Key to use for text content.

    Returns:
        List of conflict dicts with indices, texts, similarity, and patterns.
    """
    if len(evidence_list) < 2:
        return []

    texts = []
    for item in evidence_list:
        t = item.get(text_key, "")
        if not t:
            t = item.get("retrieval_text", "") or item.get("context", "")
        texts.append(t)

    model = _get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True)

    conflicts = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            # Cosine similarity
            dot = np.dot(embeddings[i], embeddings[j])
            norm = np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            sim = float(dot / norm) if norm > 0 else 0.0

            if sim >= similarity_threshold and _has_negation_conflict(texts[i], texts[j]):
                conflicts.append({
                    "evidence_i": i,
                    "evidence_j": j,
                    "similarity": round(sim, 4),
                    "text_i_snippet": texts[i][:200],
                    "text_j_snippet": texts[j][:200],
                })

    return conflicts


def add_conflict_metadata(
    evidence_list: List[Dict[str, Any]],
    similarity_threshold: float = 0.7,
    text_key: str = "text",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Add conflict flags to evidence items and return conflict pairs.

    Each evidence item gets:
      - 'has_conflict': bool
      - 'conflict_indices': list of indices it conflicts with

    Returns:
        (evidence_list with metadata, list of conflict dicts)
    """
    conflicts = detect_conflicts(evidence_list, similarity_threshold, text_key)

    # Initialize flags
    for item in evidence_list:
        item["has_conflict"] = False
        item["conflict_indices"] = []

    # Mark conflicting items
    for c in conflicts:
        i, j = c["evidence_i"], c["evidence_j"]
        evidence_list[i]["has_conflict"] = True
        evidence_list[j]["has_conflict"] = True
        evidence_list[i]["conflict_indices"].append(j)
        evidence_list[j]["conflict_indices"].append(i)

    return evidence_list, conflicts


def compute_conflict_stats(
    evidence_list: List[Dict[str, Any]],
    conflicts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute conflict statistics for a set of evidence."""
    total = len(evidence_list)
    conflicting = sum(1 for e in evidence_list if e.get("has_conflict", False))

    return {
        "total_evidence": total,
        "num_conflicts": len(conflicts),
        "conflicting_items": conflicting,
        "conflict_fraction": conflicting / total if total > 0 else 0.0,
        "conflict_pairs": conflicts,
    }
