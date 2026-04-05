"""Output data models for the Judge module.

Defines typed contracts for trust scores, judge verdicts,
experiment results, and evaluation reports.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TrustScore(BaseModel):
    """Composite trust score built from three sub-signals."""

    overall: float = Field(ge=0.0, le=1.0, description="Weighted combination of all sub-signals")
    agent_agreement: float = Field(ge=0.0, le=1.0, description="1.0 when agents agree, 0.0 when opposed")
    reasoning_consistency: float = Field(ge=0.0, le=1.0, description="Semantic similarity of reasoning across rounds")
    confidence_stability: float = Field(ge=0.0, le=1.0, description="How stable each agent's confidence was")
    breakdown: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw signals and weights for debugging / ablation",
    )


class JudgeVerdict(BaseModel):
    """Final output of the judge agent for a single question."""

    question: str
    final_answer: Literal["yes", "no", "maybe"]
    explanation: str = Field(description="Judge's synthesised justification citing evidence and arguments")
    trust: TrustScore
    debate_summary: str = Field(default="", description="2-3 sentence human-readable debate summary")


class ExperimentResult(BaseModel):
    """Result of running one system on one question."""

    system_name: str = Field(description="e.g. baseline_rag, debate_no_trust, full_system")
    question: str
    ground_truth: str
    predicted_answer: str
    correct: bool
    trust_score: Optional[float] = Field(default=None, description="None for baselines without trust")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Aggregated evaluation metrics for one system across all questions."""

    system_name: str
    num_questions: int
    accuracy: float = Field(ge=0.0, le=1.0)
    error_rate: float = Field(ge=0.0, le=1.0, description="Fraction of wrong or 'maybe' answers")
    trust_correlation: Optional[float] = Field(
        default=None,
        description="Pearson r between trust_score and correctness (only for trust-producing systems)",
    )
    per_question_results: List[ExperimentResult] = Field(default_factory=list)
