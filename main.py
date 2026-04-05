"""
End-to-end pipeline: Retrieval → Smart Filtering → Debate → Judge → Evaluation

Modes:
  python main.py              — single-question demo
  python main.py --experiment — full experiment on 50 questions, all 4 systems
  python main.py --experiment --num_questions 100 --max_rounds 3
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import logging
import time

from src.retrieval.load_dataset import load_pubmedqa
from src.retrieval.chunk_documents import chunk_documents
from src.retrieval.create_embeddings import load_embedding_model, create_embeddings
from src.retrieval.vector_store import create_chroma_collection, store_documents
from src.retrieval.bm25_index import build_bm25_index
from src.retrieval.reranker import load_reranker
from src.retrieval.retrieve_evidence import retrieve_evidence_hybrid_reranked
from src.retrieval.temporal_filter import filter_evidence_by_recency, compute_temporal_stats
from src.retrieval.conflict_detector import add_conflict_metadata, compute_conflict_stats
from src.debate import run_debate
from src.debate.agents import create_llm, reset_token_usage, get_token_usage
from src.Judge.pipeline import run_judge_on_transcript
from src.Judge.experiments import run_all_experiments, SYSTEM_RUNNERS
from src.Judge.evaluation import (
    compute_evaluation_report,
    compute_evidence_error_rate,
    export_results_to_csv,
    export_results_to_json,
    print_comparison_table,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Retrieval Pipeline Setup ──────────────────────────────────────

def build_pipeline(limit: int = 200):
    """Build the full retrieval pipeline (shared across modes)."""
    print("Loading PubMedQA dataset...")
    dataset = load_pubmedqa(limit=limit)
    documents = chunk_documents(dataset)
    print(f"Loaded {len(documents)} documents (with temporal metadata)")

    print("Creating embeddings...")
    model = load_embedding_model()
    embeddings = create_embeddings(model, documents)

    print("Building BM25 index and reranker...")
    bm25 = build_bm25_index(documents)
    reranker = load_reranker()

    collection = create_chroma_collection("pubmedqa_main", "./chroma_store_main")
    if collection.count() == 0:
        store_documents(collection, documents, embeddings=embeddings)
    print(f"ChromaDB ready: {collection.count()} docs")

    return collection, model, bm25, documents, reranker


def retrieve_and_filter(collection, model, bm25, documents, reranker, query, top_k=5):
    """Retrieve evidence with temporal + conflict filtering."""
    evidence = retrieve_evidence_hybrid_reranked(
        collection, model, bm25, documents, reranker,
        query=query, top_k=top_k, candidate_k=15,
    )

    # Apply temporal filtering (flag mode — keeps all, adds metadata)
    evidence = filter_evidence_by_recency(evidence, recency_threshold=2010, mode="flag")

    # Apply conflict detection
    evidence, conflicts = add_conflict_metadata(evidence)

    return evidence, conflicts


# ── Mode 1: Single-Question Demo ─────────────────────────────────

def run_single_demo(args):
    """Run full pipeline on a single question to demo the system."""
    collection, model, bm25, documents, reranker = build_pipeline(limit=200)

    question = documents[0]["question"]
    ground_truth = documents[0]["final_decision"]

    print(f"\nQuestion: {question}")
    print(f"Ground truth: {ground_truth}")

    # Retrieve + filter
    print("\nRetrieving and filtering evidence...")
    evidence, conflicts = retrieve_and_filter(
        collection, model, bm25, documents, reranker, question
    )

    # Temporal stats
    temporal = compute_temporal_stats(evidence)
    print(f"  Evidence items: {temporal['total']}")
    print(f"  Outdated: {temporal['outdated']} ({temporal['outdated_fraction']:.0%})")
    print(f"  Estimated years: {temporal['years']}")

    # Conflict stats
    conflict = compute_conflict_stats(evidence, conflicts)
    print(f"  Conflicts detected: {conflict['num_conflicts']}")

    # Evidence error rate
    error_info = compute_evidence_error_rate(evidence)
    print(f"  Evidence error rate: {error_info['evidence_error_rate']:.2%}")

    # Debate
    reset_token_usage()
    print("\nRunning debate (2 rounds)...")
    transcript = run_debate(question=question, evidence=evidence, max_rounds=2)

    # Print debate
    print("\n" + "=" * 60)
    print("DEBATE TRANSCRIPT")
    print("=" * 60)
    for msg in transcript.messages:
        print(f"\n[Round {msg.round}] {msg.agent.upper()}")
        print(f"  Position:   {msg.position}")
        print(f"  Confidence: {msg.confidence}")
        print(f"  Evidence:   {msg.evidence_cited}")
        print(f"  Reasoning:  {msg.reasoning[:300]}...")

    # Judge pipeline
    print("\nRunning judge pipeline...")
    verdict, trust, result = run_judge_on_transcript(
        transcript=transcript, ground_truth=ground_truth,
    )

    tokens = get_token_usage()

    print("\n" + "=" * 60)
    print("JUDGE VERDICT")
    print("=" * 60)
    print(f"Final Answer : {verdict.final_answer}")
    print(f"Explanation  : {verdict.explanation}")
    print(f"Debate Summary: {verdict.debate_summary}")
    print(f"\nTrust Score  : {trust.overall:.3f}")
    print(f"  Agreement       : {trust.agent_agreement:.2f}")
    print(f"  Consistency     : {trust.reasoning_consistency:.2f}")
    print(f"  Stability       : {trust.confidence_stability:.2f}")
    print(f"\nToken Usage  : {tokens['total_tokens']} total ({tokens['num_calls']} LLM calls)")
    if result:
        print(f"\nCorrect: {result.correct}")
    print("=" * 60)

    # Save
    if result:
        report = compute_evaluation_report([result], system_name="full_system")
        csv_path = export_results_to_csv([report])
        json_path = export_results_to_json([report])
        print(f"\nArtifacts: {csv_path}, {json_path}")


# ── Mode 2: Full Experiment ──────────────────────────────────────

def _majority_vote(pos_a, pos_b, conf_a, conf_b):
    """Simple majority vote between two doctors."""
    if pos_a == pos_b:
        return pos_a
    return pos_a if conf_a >= conf_b else pos_b


def run_experiment(args):
    """Run all 4 systems on N questions and produce comparison tables.

    Key design: debate runs ONCE per question and is shared between
    debate_no_trust and full_system, ensuring a fair comparison.
    """
    num_q = args.num_questions
    max_rounds = args.max_rounds
    limit = max(num_q + 50, 200)

    collection, model, bm25, documents, reranker = build_pipeline(limit=limit)
    llm = create_llm()

    # Build dataset items with evidence for each question
    print(f"\nPreparing {num_q} questions with evidence...")
    dataset_items = []
    for i in range(min(num_q, len(documents))):
        question = documents[i]["question"]
        ground_truth = documents[i]["final_decision"]

        evidence, _ = retrieve_and_filter(
            collection, model, bm25, documents, reranker, question, top_k=5
        )

        dataset_items.append({
            "question": question,
            "final_decision": ground_truth,
            "evidence": evidence,
        })

    print(f"Prepared {len(dataset_items)} questions with retrieved evidence")
    print(f"Max rounds: {max_rounds}")

    from src.Judge.experiments import (
        _run_baseline_no_retrieval, _run_baseline_rag, _parse_baseline_answer,
        BASELINE_NO_RETRIEVAL_SYSTEM, BASELINE_NO_RETRIEVAL_USER,
        BASELINE_RAG_SYSTEM, BASELINE_RAG_USER,
    )
    from src.Judge.trust import compute_trust_score
    from src.Judge.judge_agent import run_judge
    from src.Judge.models import ExperimentResult
    from src.debate.prompts import format_evidence_for_prompt
    from langchain_core.messages import HumanMessage, SystemMessage

    all_results = {
        "baseline_no_retrieval": [],
        "baseline_rag": [],
        "debate_no_trust": [],
        "full_system": [],
    }
    token_usage_per_system = {}

    # ── 1. Baselines (no debate needed) ──
    print(f"\n{'='*60}")
    print("Running baseline_no_retrieval...")
    print(f"{'='*60}")
    reset_token_usage()
    start = time.time()
    for idx, item in enumerate(dataset_items):
        try:
            result = _run_baseline_no_retrieval(item["question"], item["evidence"], item["final_decision"], llm)
            all_results["baseline_no_retrieval"].append(result)
            status = "CORRECT" if result.correct else "WRONG"
            print(f"  [{idx+1}/{len(dataset_items)}] {status} — predicted={result.predicted_answer}, truth={result.ground_truth}")
        except Exception as e:
            logger.error("  Q%d FAILED: %s", idx+1, e)
            all_results["baseline_no_retrieval"].append(ExperimentResult(
                system_name="baseline_no_retrieval", question=item["question"],
                ground_truth=item["final_decision"], predicted_answer="maybe",
                correct=("maybe" == item["final_decision"]), metadata={"error": str(e)},
            ))
    elapsed = time.time() - start
    token_usage_per_system["baseline_no_retrieval"] = get_token_usage()
    print(f"  Done in {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print("Running baseline_rag...")
    print(f"{'='*60}")
    reset_token_usage()
    start = time.time()
    for idx, item in enumerate(dataset_items):
        try:
            result = _run_baseline_rag(item["question"], item["evidence"], item["final_decision"], llm)
            all_results["baseline_rag"].append(result)
            status = "CORRECT" if result.correct else "WRONG"
            print(f"  [{idx+1}/{len(dataset_items)}] {status} — predicted={result.predicted_answer}, truth={result.ground_truth}")
        except Exception as e:
            logger.error("  Q%d FAILED: %s", idx+1, e)
            all_results["baseline_rag"].append(ExperimentResult(
                system_name="baseline_rag", question=item["question"],
                ground_truth=item["final_decision"], predicted_answer="maybe",
                correct=("maybe" == item["final_decision"]), metadata={"error": str(e)},
            ))
    elapsed = time.time() - start
    token_usage_per_system["baseline_rag"] = get_token_usage()
    print(f"  Done in {elapsed:.1f}s")

    # ── 2. Debate (run ONCE, score both ways) ──
    print(f"\n{'='*60}")
    print("Running debate (shared between debate_no_trust + full_system)...")
    print(f"{'='*60}")
    reset_token_usage()
    start = time.time()

    for idx, item in enumerate(dataset_items):
        question = item["question"]
        evidence = item["evidence"]
        ground_truth = item["final_decision"]

        try:
            # Run debate ONCE
            transcript = run_debate(question=question, evidence=evidence, max_rounds=max_rounds)

            # Score method 1: majority vote (debate_no_trust)
            mv_answer = _majority_vote(
                transcript.doctor_a_final_position,
                transcript.doctor_b_final_position,
                transcript.doctor_a_final_confidence,
                transcript.doctor_b_final_confidence,
            )
            all_results["debate_no_trust"].append(ExperimentResult(
                system_name="debate_no_trust", question=question,
                ground_truth=ground_truth, predicted_answer=mv_answer,
                correct=(mv_answer == ground_truth), trust_score=None,
                metadata={
                    "doctor_a_final": transcript.doctor_a_final_position,
                    "doctor_b_final": transcript.doctor_b_final_position,
                },
            ))

            # Score method 2: trust-aware judge (full_system)
            trust = compute_trust_score(transcript)
            verdict = run_judge(transcript, trust, llm=llm)
            all_results["full_system"].append(ExperimentResult(
                system_name="full_system", question=question,
                ground_truth=ground_truth, predicted_answer=verdict.final_answer,
                correct=(verdict.final_answer == ground_truth),
                trust_score=trust.overall,
                metadata={
                    "doctor_a_final": transcript.doctor_a_final_position,
                    "doctor_b_final": transcript.doctor_b_final_position,
                    "trust_breakdown": trust.breakdown,
                    "judge_explanation": verdict.explanation[:500],
                },
            ))

            mv_status = "CORRECT" if mv_answer == ground_truth else "WRONG"
            fs_status = "CORRECT" if verdict.final_answer == ground_truth else "WRONG"
            print(f"  [{idx+1}/{len(dataset_items)}] debate_no_trust={mv_status}({mv_answer}) | full_system={fs_status}({verdict.final_answer}) | truth={ground_truth}")

        except Exception as e:
            logger.error("  Q%d FAILED: %s", idx+1, e)
            for sys_name in ("debate_no_trust", "full_system"):
                all_results[sys_name].append(ExperimentResult(
                    system_name=sys_name, question=question,
                    ground_truth=ground_truth, predicted_answer="maybe",
                    correct=("maybe" == ground_truth), metadata={"error": str(e)},
                ))

    elapsed = time.time() - start
    debate_tokens = get_token_usage()
    token_usage_per_system["debate_no_trust"] = debate_tokens
    token_usage_per_system["full_system"] = debate_tokens
    print(f"  Debate done in {elapsed:.1f}s — {debate_tokens['total_tokens']} tokens, {debate_tokens['num_calls']} calls")

    # Compute reports
    reports = []
    for system_name in all_results:
        report = compute_evaluation_report(all_results[system_name], system_name=system_name)
        reports.append(report)

    # Print comparison
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    print_comparison_table(reports, token_usage=token_usage_per_system)

    # Evidence quality stats (aggregate across all questions)
    print("\n" + "=" * 60)
    print("EVIDENCE QUALITY (across all questions)")
    print("=" * 60)
    all_evidence = []
    for item in dataset_items:
        all_evidence.extend(item["evidence"])
    if all_evidence:
        eq = compute_evidence_error_rate(all_evidence)
        print(f"  Total evidence items: {eq['total_evidence']}")
        print(f"  Outdated fraction: {eq['temporal']['outdated_fraction']:.2%}")
        print(f"  Conflict fraction: {eq['conflict']['conflict_fraction']:.2%}")
        print(f"  Combined evidence error rate: {eq['evidence_error_rate']:.2%}")

    # Export
    csv_path = export_results_to_csv(reports, "outputs/experiment_results.csv")
    json_path = export_results_to_json(reports, "outputs/experiment_results.json")
    print(f"\nExported: {csv_path}, {json_path}")
    print("=" * 60)


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Medical Debate System")
    parser.add_argument("--experiment", action="store_true", help="Run full experiment mode")
    parser.add_argument("--num_questions", type=int, default=50, help="Number of questions (experiment mode)")
    parser.add_argument("--max_rounds", type=int, default=2, help="Debate rounds")
    args = parser.parse_args()

    if args.experiment:
        run_experiment(args)
    else:
        run_single_demo(args)


if __name__ == "__main__":
    main()
