"""Judge module — trust-aware medical debate adjudication.

Public API:
    run_judge_pipeline()       — full pipeline: debate → trust → judge
    run_judge_on_transcript()  — judge + trust on a pre-existing transcript
    run_judge()                — judge agent only
    compute_trust_score()      — trust score calculation only
    run_all_experiments()      — run baseline vs proposed experiments
    compute_evaluation_report() — compute metrics from experiment results
"""

from src.Judge.models import (
    EvaluationReport,
    ExperimentResult,
    JudgeVerdict,
    TrustScore,
)
from src.Judge.pipeline import run_judge_on_transcript, run_judge_pipeline
from src.Judge.judge_agent import run_judge
from src.Judge.trust import compute_trust_score
from src.Judge.experiments import run_all_experiments, run_single_experiment
from src.Judge.evaluation import (
    compute_evaluation_report,
    export_results_to_csv,
    export_results_to_json,
    print_comparison_table,
)
