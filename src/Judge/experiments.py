"""Experiment runner — compare baseline RAG, debate-only, and full system.

Three systems are evaluated on the same set of PubMedQA questions:
  1. baseline_rag      — single LLM call with retrieved evidence, no debate.
  2. debate_no_trust   — full debate, majority-vote judge (no trust scoring).
  3. full_system       — full debate + trust-aware judge (proposed approach).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.debate.agents import create_llm
from src.debate.graph import run_debate
from src.debate.prompts import format_evidence_for_prompt
from src.Judge.judge_agent import run_judge
from src.Judge.models import ExperimentResult
from src.Judge.trust import compute_trust_score

logger = logging.getLogger(__name__)

# ── Baseline RAG Prompt ─────────────────────────────────────────

BASELINE_RAG_SYSTEM = """\
You are a medical expert answering biomedical research questions.
Based ONLY on the provided evidence, determine whether the answer
to the question is yes, no, or maybe.

You MUST respond in EXACTLY this format:

ANSWER: yes
EXPLANATION: <brief justification citing evidence>

Rules:
- ANSWER must be exactly one of: yes, no, maybe
"""

BASELINE_RAG_USER = """\
MEDICAL QUESTION: {question}

RETRIEVED EVIDENCE:
{formatted_evidence}

Based on the evidence above, what is your answer?"""


# ── System Runners ──────────────────────────────────────────────


def _run_baseline_rag(
    question: str,
    evidence: List[dict],
    ground_truth: str,
    llm,
) -> ExperimentResult:
    """Baseline: single LLM call with evidence, no debate."""
    formatted = format_evidence_for_prompt(evidence)
    messages = [
        SystemMessage(content=BASELINE_RAG_SYSTEM),
        HumanMessage(
            content=BASELINE_RAG_USER.format(
                question=question, formatted_evidence=formatted
            )
        ),
    ]

    response = llm.invoke(messages)
    answer = _parse_baseline_answer(response.content)

    return ExperimentResult(
        system_name="baseline_rag",
        question=question,
        ground_truth=ground_truth,
        predicted_answer=answer,
        correct=(answer == ground_truth),
        trust_score=None,
        metadata={"raw_response": response.content[:500]},
    )


def _parse_baseline_answer(text: str) -> str:
    """Extract the answer from a baseline RAG response."""
    match = re.search(r"ANSWER:\s*(yes|no|maybe)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    # Fallback: look for the word in the first line
    first_line = text.strip().split("\n")[0].lower()
    for option in ("yes", "no", "maybe"):
        if option in first_line:
            return option
    return "maybe"


def _run_debate_no_trust(
    question: str,
    evidence: List[dict],
    ground_truth: str,
    max_rounds: int,
    llm,
) -> ExperimentResult:
    """Debate pipeline with majority-vote judge (no trust scoring)."""
    transcript = run_debate(question=question, evidence=evidence, max_rounds=max_rounds)

    # Majority vote: if both agree, take that; otherwise "maybe"
    answer = _majority_vote(
        transcript.doctor_a_final_position,
        transcript.doctor_b_final_position,
        transcript.doctor_a_final_confidence,
        transcript.doctor_b_final_confidence,
    )

    return ExperimentResult(
        system_name="debate_no_trust",
        question=question,
        ground_truth=ground_truth,
        predicted_answer=answer,
        correct=(answer == ground_truth),
        trust_score=None,
        metadata={
            "doctor_a_final": transcript.doctor_a_final_position,
            "doctor_b_final": transcript.doctor_b_final_position,
            "num_rounds": transcript.num_rounds,
        },
    )


def _majority_vote(
    pos_a: str, pos_b: str, conf_a: float, conf_b: float
) -> str:
    """Simple majority vote between two doctors.

    If they agree, return their shared position.
    If they disagree, return the position of the more confident doctor.
    """
    if pos_a == pos_b:
        return pos_a
    # Disagreement — pick the more confident doctor
    return pos_a if conf_a >= conf_b else pos_b


def _run_full_system(
    question: str,
    evidence: List[dict],
    ground_truth: str,
    max_rounds: int,
    llm,
) -> ExperimentResult:
    """Full system: debate + trust-aware judge."""
    transcript = run_debate(question=question, evidence=evidence, max_rounds=max_rounds)
    trust = compute_trust_score(transcript)
    verdict = run_judge(transcript, trust, llm=llm)

    return ExperimentResult(
        system_name="full_system",
        question=question,
        ground_truth=ground_truth,
        predicted_answer=verdict.final_answer,
        correct=(verdict.final_answer == ground_truth),
        trust_score=trust.overall,
        metadata={
            "doctor_a_final": transcript.doctor_a_final_position,
            "doctor_b_final": transcript.doctor_b_final_position,
            "trust_breakdown": trust.breakdown,
            "judge_explanation": verdict.explanation[:500],
            "num_rounds": transcript.num_rounds,
        },
    )


# ── Public API ──────────────────────────────────────────────────

SYSTEM_RUNNERS = {
    "baseline_rag": _run_baseline_rag,
    "debate_no_trust": _run_debate_no_trust,
    "full_system": _run_full_system,
}


def run_single_experiment(
    question: str,
    evidence: List[dict],
    ground_truth: str,
    system_name: str,
    max_rounds: int = 2,
    llm=None,
) -> ExperimentResult:
    """Run one system on one question.

    Args:
        question: The biomedical question.
        evidence: Retrieved evidence dicts.
        ground_truth: The correct answer (yes/no/maybe).
        system_name: One of 'baseline_rag', 'debate_no_trust', 'full_system'.
        max_rounds: Debate rounds (ignored for baseline_rag).
        llm: Optional LLM instance.

    Returns:
        An ExperimentResult for this question + system pair.
    """
    if llm is None:
        llm = create_llm()

    if system_name == "baseline_rag":
        return _run_baseline_rag(question, evidence, ground_truth, llm)
    elif system_name == "debate_no_trust":
        return _run_debate_no_trust(question, evidence, ground_truth, max_rounds, llm)
    elif system_name == "full_system":
        return _run_full_system(question, evidence, ground_truth, max_rounds, llm)
    else:
        raise ValueError(f"Unknown system: {system_name}. Choose from {list(SYSTEM_RUNNERS)}")


def run_all_experiments(
    dataset: List[Dict[str, Any]],
    systems: Optional[List[str]] = None,
    num_questions: int = 50,
    max_rounds: int = 2,
    llm=None,
) -> Dict[str, List[ExperimentResult]]:
    """Run all systems on the same set of questions.

    Args:
        dataset: List of PubMedQA-style dicts with 'question',
                 'final_decision', and evidence fields.
        systems: Which systems to run (default: all three).
        num_questions: How many questions to evaluate.
        max_rounds: Debate rounds for debate-based systems.
        llm: Optional shared LLM instance.

    Returns:
        Dict mapping system_name → list of ExperimentResult.
    """
    if llm is None:
        llm = create_llm()

    if systems is None:
        systems = list(SYSTEM_RUNNERS)

    results: Dict[str, List[ExperimentResult]] = {s: [] for s in systems}
    questions_to_run = dataset[:num_questions]

    for idx, item in enumerate(questions_to_run):
        question = item["question"]
        ground_truth = item["final_decision"]
        evidence = item.get("evidence", [])

        logger.info(
            "Question %d/%d: %s",
            idx + 1,
            len(questions_to_run),
            question[:80],
        )

        for system_name in systems:
            try:
                result = run_single_experiment(
                    question=question,
                    evidence=evidence,
                    ground_truth=ground_truth,
                    system_name=system_name,
                    max_rounds=max_rounds,
                    llm=llm,
                )
                results[system_name].append(result)
                logger.info(
                    "  %s → %s (correct=%s)",
                    system_name,
                    result.predicted_answer,
                    result.correct,
                )
            except Exception as e:
                logger.error("  %s FAILED: %s", system_name, e)
                results[system_name].append(
                    ExperimentResult(
                        system_name=system_name,
                        question=question,
                        ground_truth=ground_truth,
                        predicted_answer="maybe",
                        correct=("maybe" == ground_truth),
                        trust_score=None,
                        metadata={"error": str(e)},
                    )
                )

    return results
