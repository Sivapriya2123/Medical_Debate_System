"""Evaluation metrics — accuracy, error rate, trust-accuracy correlation.

Consumes lists of ExperimentResult and produces EvaluationReport objects
that can be printed as comparison tables or exported to CSV / JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from scipy.stats import pearsonr

from src.Judge.models import EvaluationReport, ExperimentResult
from src.retrieval.temporal_filter import compute_temporal_stats
from src.retrieval.conflict_detector import add_conflict_metadata, compute_conflict_stats

logger = logging.getLogger(__name__)


# ── Evidence Quality Metrics ──────────────────────────────────────


def compute_evidence_error_rate(
    evidence_list: List[dict],
    recency_threshold: int = 2010,
    similarity_threshold: float = 0.7,
) -> Dict:
    """Compute evidence quality metrics: outdated fraction + conflict fraction.

    Args:
        evidence_list: Retrieved evidence dicts (with temporal metadata).
        recency_threshold: Year cutoff for outdated.
        similarity_threshold: Cosine sim threshold for conflict detection.

    Returns:
        Dict with temporal stats, conflict stats, and combined error rate.
    """
    temporal = compute_temporal_stats(evidence_list)

    evidence_list, conflicts = add_conflict_metadata(
        evidence_list, similarity_threshold=similarity_threshold
    )
    conflict = compute_conflict_stats(evidence_list, conflicts)

    # Combined evidence error rate: fraction of evidence that is
    # either outdated OR involved in a conflict
    total = len(evidence_list)
    problematic = set()
    if total > 0:
        for i, e in enumerate(evidence_list):
            if e.get("is_outdated", False):
                problematic.add(i)
            if e.get("has_conflict", False):
                problematic.add(i)
    combined_error = len(problematic) / total if total > 0 else 0.0

    return {
        "temporal": temporal,
        "conflict": conflict,
        "evidence_error_rate": round(combined_error, 4),
        "num_problematic": len(problematic),
        "total_evidence": total,
    }


# ── Core Metrics ────────────────────────────────────────────────


def compute_evaluation_report(
    results: List[ExperimentResult],
    system_name: str,
) -> EvaluationReport:
    """Compute aggregated evaluation metrics for one system.

    Metrics:
        - accuracy:  fraction of correct predictions
        - error_rate: fraction of wrong OR 'maybe' answers
        - trust_correlation: Pearson r(trust_score, correct) if trust available

    Args:
        results: Per-question ExperimentResults for a single system.
        system_name: Name of the system being evaluated.

    Returns:
        An EvaluationReport with all metrics populated.
    """
    if not results:
        return EvaluationReport(
            system_name=system_name,
            num_questions=0,
            accuracy=0.0,
            error_rate=1.0,
            trust_correlation=None,
            per_question_results=[],
        )

    n = len(results)
    correct_flags = [r.correct for r in results]
    accuracy = sum(correct_flags) / n

    # Error rate: wrong answers + "maybe" answers (abstentions count as errors)
    errors = sum(
        1 for r in results
        if not r.correct or r.predicted_answer == "maybe"
    )
    error_rate = errors / n

    # Trust-accuracy correlation (only if trust scores exist)
    trust_correlation = _compute_trust_correlation(results)

    return EvaluationReport(
        system_name=system_name,
        num_questions=n,
        accuracy=round(accuracy, 4),
        error_rate=round(error_rate, 4),
        trust_correlation=trust_correlation,
        per_question_results=results,
    )


def _compute_trust_correlation(
    results: List[ExperimentResult],
) -> Optional[float]:
    """Pearson correlation between trust scores and correctness.

    Returns None if fewer than 3 results have trust scores (not enough data).
    """
    paired = [
        (r.trust_score, 1.0 if r.correct else 0.0)
        for r in results
        if r.trust_score is not None
    ]

    if len(paired) < 3:
        return None

    trust_vals, correct_vals = zip(*paired)
    try:
        corr, _ = pearsonr(trust_vals, correct_vals)
        return round(float(corr), 4)
    except Exception:
        logger.warning("Could not compute Pearson correlation", exc_info=True)
        return None


# ── Comparison Table ────────────────────────────────────────────


def print_comparison_table(
    reports: List[EvaluationReport],
    token_usage: Optional[Dict[str, Dict[str, int]]] = None,
) -> str:
    """Pretty-print a markdown table comparing multiple systems.

    Args:
        reports: List of EvaluationReport objects.
        token_usage: Optional dict mapping system_name -> token stats.

    Returns the table string (also prints it to stdout).
    """
    has_tokens = token_usage is not None
    header = "| System               | Questions | Accuracy | Error Rate | Trust-Acc Corr |"
    sep    = "|----------------------|-----------|----------|------------|----------------|"
    if has_tokens:
        header += " Tokens |"
        sep    += "--------|"

    rows = [header, sep]

    for r in reports:
        corr_str = f"{r.trust_correlation:+.4f}" if r.trust_correlation is not None else "N/A"
        row = (
            f"| {r.system_name:<20} "
            f"| {r.num_questions:>9} "
            f"| {r.accuracy:>8.4f} "
            f"| {r.error_rate:>10.4f} "
            f"| {corr_str:>14} |"
        )
        if has_tokens:
            tokens = token_usage.get(r.system_name, {}).get("total_tokens", 0)
            row += f" {tokens:>6} |"
        rows.append(row)

    table = "\n".join(rows)
    print(table)
    return table


# ── Export Helpers ───────────────────────────────────────────────


def export_results_to_csv(
    reports: List[EvaluationReport],
    output_path: str = "outputs/experiment_results.csv",
) -> str:
    """Export per-question results from all systems to a CSV file.

    Returns the path to the written file.
    """
    rows = []
    for report in reports:
        for r in report.per_question_results:
            rows.append({
                "system": r.system_name,
                "question": r.question[:200],
                "ground_truth": r.ground_truth,
                "predicted": r.predicted_answer,
                "correct": r.correct,
                "trust_score": r.trust_score,
            })

    df = pd.DataFrame(rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("CSV exported to %s (%d rows)", path, len(df))
    return str(path)


def export_results_to_json(
    reports: List[EvaluationReport],
    output_path: str = "outputs/experiment_results.json",
) -> str:
    """Export full evaluation reports (with per-question detail) to JSON.

    Returns the path to the written file.
    """
    data = {
        "reports": [r.model_dump() for r in reports],
        "summary": {
            r.system_name: {
                "accuracy": r.accuracy,
                "error_rate": r.error_rate,
                "trust_correlation": r.trust_correlation,
                "num_questions": r.num_questions,
            }
            for r in reports
        },
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("JSON exported to %s", path)
    return str(path)
