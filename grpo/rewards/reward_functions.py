"""
Reward functions for GRPO training of the judge agent.

Each function takes a single debate trace and the judge's candidate output,
and returns a float reward value.

The total reward is: R_correct + R_format + R_anti_maybe + R_evidence
"""

import re
from typing import Optional


def reward_correctness(prediction: str, gold_label: str) -> float:
    """
    +1.0 if prediction matches gold label exactly, else 0.0
    This is the primary reward signal.
    """
    pred = prediction.strip().lower()
    gold = gold_label.strip().lower()
    return 1.0 if pred == gold else 0.0


def reward_format_compliance(response: str) -> float:
    """
    +0.25 if the response contains proper structured output.
    Checks for answer tag and reasoning content.

    Adjust the tag patterns to match your judge's expected output format.
    """
    has_answer = bool(re.search(r"(final\s*answer|answer)\s*[:=]\s*(yes|no|maybe)", response, re.IGNORECASE))
    has_reasoning = len(response.split()) > 20  # at least 20 words of reasoning

    if has_answer and has_reasoning:
        return 0.25
    elif has_answer:
        return 0.125
    else:
        return 0.0


def reward_anti_maybe(prediction: str, gold_label: str) -> float:
    """
    -0.3 penalty when judge predicts "maybe" but gold is "yes" or "no".
    This directly targets the over-correction problem.

    No penalty when gold IS "maybe" (that's correct behavior).
    No penalty when judge says yes/no (even if wrong -- correctness handles that).
    """
    pred = prediction.strip().lower()
    gold = gold_label.strip().lower()

    if pred == "maybe" and gold in ("yes", "no"):
        return -0.3
    return 0.0


def reward_evidence_citation(
    response: str,
    retrieved_evidence: list[str],
    min_overlap_words: int = 3
) -> float:
    """
    +0.25 if the judge's reasoning references content from retrieved evidence.
    Uses simple word overlap -- checks if key phrases from evidence appear in response.
    """
    if not retrieved_evidence:
        return 0.0

    response_lower = response.lower()
    response_words = set(response_lower.split())

    evidence_referenced = False
    for chunk in retrieved_evidence:
        chunk_words = set(chunk.lower().split())
        # Check for meaningful overlap (not just stopwords)
        overlap = chunk_words & response_words
        # Filter out very common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                     "to", "for", "of", "and", "or", "but", "not", "with", "this", "that",
                     "it", "be", "have", "has", "had", "do", "does", "did", "will", "would",
                     "can", "could", "may", "might", "shall", "should"}
        meaningful_overlap = overlap - stopwords
        if len(meaningful_overlap) >= min_overlap_words:
            evidence_referenced = True
            break

    return 0.25 if evidence_referenced else 0.0


def compute_total_reward(
    prediction: str,
    gold_label: str,
    response: str,
    retrieved_evidence: list[str],
    weights: Optional[dict] = None
) -> dict:
    """
    Compute total reward and breakdown for a single judge output.

    Returns dict with individual rewards and total.
    """
    if weights is None:
        weights = {
            "correctness": 1.0,
            "format": 1.0,
            "anti_maybe": 1.0,
            "evidence": 1.0,
        }

    r_correct = reward_correctness(prediction, gold_label) * weights["correctness"]
    r_format = reward_format_compliance(response) * weights["format"]
    r_anti_maybe = reward_anti_maybe(prediction, gold_label) * weights["anti_maybe"]
    r_evidence = reward_evidence_citation(response, retrieved_evidence) * weights["evidence"]

    total = r_correct + r_format + r_anti_maybe + r_evidence

    return {
        "total": total,
        "correctness": r_correct,
        "format": r_format,
        "anti_maybe": r_anti_maybe,
        "evidence": r_evidence,
    }


# === Unit Tests ===
def test_rewards():
    """Run basic sanity checks on reward functions."""

    # Test 1: Perfect answer
    r = compute_total_reward(
        prediction="yes",
        gold_label="yes",
        response="Based on the evidence presented about patient outcomes, the answer is clearly yes. The study demonstrates significant improvement in the treatment group compared to control.",
        retrieved_evidence=["patient outcomes showed significant improvement in treatment group"]
    )
    assert r["correctness"] == 1.0, f"Expected 1.0, got {r['correctness']}"
    assert r["anti_maybe"] == 0.0, f"Expected 0.0, got {r['anti_maybe']}"
    assert r["evidence"] == 0.25, f"Expected 0.25, got {r['evidence']}"
    print(f"Test 1 PASSED -- perfect answer: total={r['total']:.2f}")

    # Test 2: Maybe over-correction
    r = compute_total_reward(
        prediction="maybe",
        gold_label="yes",
        response="The evidence is inconclusive. Final answer: maybe",
        retrieved_evidence=["clear positive results"]
    )
    assert r["correctness"] == 0.0
    assert r["anti_maybe"] == -0.3
    print(f"Test 2 PASSED -- maybe over-correction: total={r['total']:.2f}")

    # Test 3: Correct maybe
    r = compute_total_reward(
        prediction="maybe",
        gold_label="maybe",
        response="The study results are mixed and do not provide a definitive answer either way. Final answer: maybe. The evidence from the clinical trial shows conflicting outcomes across different patient subgroups.",
        retrieved_evidence=["conflicting outcomes across patient subgroups"]
    )
    assert r["correctness"] == 1.0
    assert r["anti_maybe"] == 0.0  # no penalty -- gold IS maybe
    print(f"Test 3 PASSED -- correct maybe: total={r['total']:.2f}")

    # Test 4: Wrong answer, no maybe penalty
    r = compute_total_reward(
        prediction="no",
        gold_label="yes",
        response="No evidence supports this. Answer: no",
        retrieved_evidence=["strong evidence supports the hypothesis"]
    )
    assert r["correctness"] == 0.0
    assert r["anti_maybe"] == 0.0  # no penalty -- didn't say maybe
    print(f"Test 4 PASSED -- wrong but not maybe: total={r['total']:.2f}")

    print("\nAll reward function tests passed!")


if __name__ == "__main__":
    test_rewards()
