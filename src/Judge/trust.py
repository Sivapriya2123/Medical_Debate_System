"""Trust score calculation from debate transcripts.

Three independently computable sub-signals are combined into a single
composite trust score:
  1. Agent agreement     (weight 0.40) — do the doctors converge?
  2. Reasoning consistency (weight 0.35) — is each doctor self-consistent?
  3. Confidence stability  (weight 0.25) — are confidence values stable?
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.Judge.models import TrustScore

if TYPE_CHECKING:
    from src.debate.models import AgentMessage, DebateTranscript

logger = logging.getLogger(__name__)

# ── Weights (configurable for ablation) ─────────────────────────

DEFAULT_WEIGHTS: Dict[str, float] = {
    "agreement": 0.40,
    "consistency": 0.35,
    "stability": 0.25,
}

# ── Lazy-loaded embedding model ─────────────────────────────────

_embedding_model: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    """Load the sentence-transformer model once and cache it."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# ── Sub-signal 1: Agent Agreement ───────────────────────────────

def compute_agent_agreement(transcript: DebateTranscript) -> float:
    """Score how much the two doctors agree on the final answer.

    Returns:
        1.0 — full agreement (same position)
        0.5 — partial (one says 'maybe')
        0.0 — direct contradiction (yes vs no)
    """
    pos_a = transcript.doctor_a_final_position
    pos_b = transcript.doctor_b_final_position

    if pos_a == pos_b:
        return 1.0
    if "maybe" in (pos_a, pos_b):
        return 0.5
    return 0.0


# ── Sub-signal 2: Reasoning Consistency ─────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def _agent_reasoning_consistency(
    reasoning_texts: List[str],
    model: SentenceTransformer,
) -> float:
    """Average pairwise cosine similarity of one agent's reasoning across rounds."""
    if len(reasoning_texts) < 2:
        return 1.0  # single round — perfectly consistent by definition

    embeddings = model.encode(reasoning_texts, convert_to_numpy=True)

    similarities = []
    for i in range(len(embeddings) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    return float(np.mean(similarities))


def compute_reasoning_consistency(messages: List[AgentMessage]) -> float:
    """Mean reasoning consistency across both agents.

    Uses embedding cosine similarity to check whether each doctor's
    reasoning stays coherent across debate rounds.
    """
    model = _get_embedding_model()

    agent_scores = []
    for agent_name in ("doctor_a", "doctor_b"):
        texts = [m.reasoning for m in messages if m.agent == agent_name]
        if texts:
            agent_scores.append(_agent_reasoning_consistency(texts, model))

    if not agent_scores:
        return 0.5  # no messages — neutral score

    return float(np.mean(agent_scores))


# ── Sub-signal 3: Confidence Stability ──────────────────────────

def compute_confidence_stability(messages: List[AgentMessage]) -> float:
    """Score how stable each agent's confidence was across rounds.

    Low standard deviation → high stability.
    Returns 1.0 - mean(std_dev_a, std_dev_b), clamped to [0, 1].
    """
    stdevs = []
    for agent_name in ("doctor_a", "doctor_b"):
        confs = [m.confidence for m in messages if m.agent == agent_name]
        if len(confs) >= 2:
            stdevs.append(float(np.std(confs)))

    if not stdevs:
        return 1.0  # single round — perfectly stable

    mean_std = float(np.mean(stdevs))
    # Confidence is 0-1, so std is at most 0.5; normalise and invert.
    stability = max(0.0, min(1.0, 1.0 - 2.0 * mean_std))
    return round(stability, 4)


# ── Composite Trust Score ───────────────────────────────────────

def compute_trust_score(
    transcript: DebateTranscript,
    weights: Dict[str, float] | None = None,
) -> TrustScore:
    """Compute the composite trust score for a completed debate.

    Args:
        transcript: The full debate transcript from the debate module.
        weights: Optional override for sub-signal weights (must sum to 1.0).

    Returns:
        A TrustScore with overall score and individual sub-signals.
    """
    w = weights or DEFAULT_WEIGHTS

    agreement = compute_agent_agreement(transcript)
    consistency = compute_reasoning_consistency(transcript.messages)
    stability = compute_confidence_stability(transcript.messages)

    overall = (
        w["agreement"] * agreement
        + w["consistency"] * consistency
        + w["stability"] * stability
    )

    return TrustScore(
        overall=round(overall, 4),
        agent_agreement=round(agreement, 4),
        reasoning_consistency=round(consistency, 4),
        confidence_stability=round(stability, 4),
        breakdown={
            "weights": w,
            "raw": {
                "agreement": agreement,
                "consistency": consistency,
                "stability": stability,
            },
        },
    )
