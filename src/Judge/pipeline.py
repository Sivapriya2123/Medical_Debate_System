"""Pipeline entrypoint — wires debate → trust → judge for a single question.

This is the main integration point that `main.py` and the experiments module
call. It orchestrates the full flow and returns all outputs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.debate.graph import run_debate
from src.debate.models import DebateTranscript
from src.Judge.judge_agent import run_judge
from src.Judge.models import ExperimentResult, JudgeVerdict, TrustScore
from src.Judge.trust import compute_trust_score

logger = logging.getLogger(__name__)


def run_judge_pipeline(
    question: str,
    evidence: List[dict],
    ground_truth: Optional[str] = None,
    max_rounds: int = 2,
    llm=None,
) -> Tuple[JudgeVerdict, TrustScore, Optional[ExperimentResult]]:
    """Full pipeline for a single question.

    Steps:
        1. Run multi-agent debate (Person 2's system).
        2. Compute trust score from the debate transcript.
        3. Run judge agent to produce the final answer.
        4. Package as ExperimentResult if ground_truth is provided.

    Args:
        question: The biomedical yes/no/maybe question.
        evidence: List of evidence dicts from the retrieval pipeline.
        ground_truth: Optional correct answer for evaluation.
        max_rounds: Number of debate rounds (default 2).
        llm: Optional LLM instance; creates a default one if not provided.

    Returns:
        (JudgeVerdict, TrustScore, ExperimentResult or None)
    """
    # 1. Debate
    logger.info("Running debate (%d rounds)...", max_rounds)
    transcript = run_debate(question=question, evidence=evidence, max_rounds=max_rounds)

    # 2. Trust
    logger.info("Computing trust score...")
    trust = compute_trust_score(transcript)

    # 3. Judge
    logger.info("Running judge agent...")
    verdict = run_judge(transcript, trust, llm=llm)

    # 4. Experiment result (only if ground truth is available)
    experiment = None
    if ground_truth is not None:
        experiment = ExperimentResult(
            system_name="full_system",
            question=question,
            ground_truth=ground_truth,
            predicted_answer=verdict.final_answer,
            correct=(verdict.final_answer == ground_truth),
            trust_score=trust.overall,
            metadata={
                "num_rounds": max_rounds,
                "doctor_a_final": transcript.doctor_a_final_position,
                "doctor_b_final": transcript.doctor_b_final_position,
            },
        )

    return verdict, trust, experiment


def run_judge_on_transcript(
    transcript: DebateTranscript,
    ground_truth: Optional[str] = None,
    llm=None,
) -> Tuple[JudgeVerdict, TrustScore, Optional[ExperimentResult]]:
    """Run judge pipeline on an already-completed debate transcript.

    Useful when the debate has already been run (e.g. in experiments)
    and you just need the judge + trust scoring.

    Args:
        transcript: A completed DebateTranscript.
        ground_truth: Optional correct answer for evaluation.
        llm: Optional LLM instance.

    Returns:
        (JudgeVerdict, TrustScore, ExperimentResult or None)
    """
    trust = compute_trust_score(transcript)
    verdict = run_judge(transcript, trust, llm=llm)

    experiment = None
    if ground_truth is not None:
        experiment = ExperimentResult(
            system_name="full_system",
            question=transcript.question,
            ground_truth=ground_truth,
            predicted_answer=verdict.final_answer,
            correct=(verdict.final_answer == ground_truth),
            trust_score=trust.overall,
            metadata={
                "num_rounds": transcript.num_rounds,
                "doctor_a_final": transcript.doctor_a_final_position,
                "doctor_b_final": transcript.doctor_b_final_position,
            },
        )

    return verdict, trust, experiment
