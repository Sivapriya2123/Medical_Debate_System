"""Run a single custom question through the full pipeline with detailed output.
Usage: python run_single_question.py "Does long-term aspirin use reduce the risk of colorectal cancer?"
"""

from dotenv import load_dotenv
load_dotenv()

import sys

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
from src.debate.agents import reset_token_usage, get_token_usage
from src.Judge.pipeline import run_judge_on_transcript
from src.Judge.evaluation import compute_evidence_error_rate


def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "Does long-term aspirin use reduce the risk of colorectal cancer?"

    print("=" * 70)
    print("TRUST-AWARE MULTI-AGENT MEDICAL DEBATE SYSTEM")
    print("=" * 70)
    print(f"\nQUESTION: {question}")
    print("=" * 70)

    # ── 1. Build retrieval pipeline ──
    print("\n[STAGE 1] EVIDENCE RETRIEVAL")
    print("-" * 70)
    print("Loading PubMedQA dataset and building retrieval pipeline...")
    dataset = load_pubmedqa(limit=1000)
    documents = chunk_documents(dataset)
    model = load_embedding_model()
    embeddings = create_embeddings(model, documents)
    bm25 = build_bm25_index(documents)
    reranker = load_reranker()
    collection = create_chroma_collection("pubmedqa_demo", "./chroma_store_demo")
    if collection.count() == 0:
        store_documents(collection, documents, embeddings=embeddings)
    print(f"  Indexed {collection.count()} documents in ChromaDB")

    # ── 2. Retrieve + filter evidence ──
    print("\nRetrieving top-5 evidence passages...")
    evidence = retrieve_evidence_hybrid_reranked(
        collection, model, bm25, documents, reranker,
        query=question, top_k=5, candidate_k=15,
    )
    evidence = filter_evidence_by_recency(evidence, recency_threshold=2010, mode="flag")
    evidence, conflicts = add_conflict_metadata(evidence)

    print(f"\n  Retrieved {len(evidence)} evidence passages:")
    for i, e in enumerate(evidence, 1):
        year = e.get("estimated_year", "?")
        outdated = " [OUTDATED]" if e.get("is_outdated") else ""
        conflict = " [CONFLICT]" if e.get("has_conflict") else ""
        text = e.get("text", e.get("metadata", {}).get("context", ""))[:200]
        score = e.get("rerank_score", e.get("hybrid_score", 0))
        print(f"\n  [Evidence {i}] (year~{year}, score={score:.4f}){outdated}{conflict}")
        print(f"    {text}...")

    # Temporal + conflict stats
    temporal = compute_temporal_stats(evidence)
    conflict_stats = compute_conflict_stats(evidence, conflicts)
    eq = compute_evidence_error_rate(evidence)
    print(f"\n  EVIDENCE QUALITY:")
    print(f"    Outdated: {temporal['outdated']}/{temporal['total']} ({temporal['outdated_fraction']:.0%})")
    print(f"    Conflicts: {conflict_stats['num_conflicts']}")
    print(f"    Evidence Error Rate: {eq['evidence_error_rate']:.2%}")

    # ── 3. Debate ──
    print("\n" + "=" * 70)
    print("[STAGE 2] MULTI-AGENT DEBATE (2 rounds)")
    print("-" * 70)
    reset_token_usage()
    transcript = run_debate(question=question, evidence=evidence, max_rounds=2)

    for msg in transcript.messages:
        agent_label = "DOCTOR A (Conservative)" if msg.agent == "doctor_a" else "DOCTOR B (Generalist)"
        print(f"\n  [{agent_label} | Round {msg.round}]")
        print(f"  Position:   {msg.position.upper()}")
        print(f"  Confidence: {msg.confidence}")
        print(f"  Evidence:   {', '.join(msg.evidence_cited) if msg.evidence_cited else 'none'}")
        print(f"  Reasoning:  {msg.reasoning}")

    print(f"\n  FINAL POSITIONS:")
    print(f"    Doctor A: {transcript.doctor_a_final_position.upper()} (confidence: {transcript.doctor_a_final_confidence:.2f})")
    print(f"    Doctor B: {transcript.doctor_b_final_position.upper()} (confidence: {transcript.doctor_b_final_confidence:.2f})")

    # ── 4. Trust score + Judge ──
    print("\n" + "=" * 70)
    print("[STAGE 3] TRUST SCORE & JUDGE VERDICT")
    print("-" * 70)
    verdict, trust, _ = run_judge_on_transcript(transcript=transcript)
    tokens = get_token_usage()

    print(f"\n  TRUST SCORE: {trust.overall:.3f}")
    print(f"    Agent Agreement:       {trust.agent_agreement:.2f}  (weight: 0.40)")
    print(f"    Reasoning Consistency: {trust.reasoning_consistency:.2f}  (weight: 0.35)")
    print(f"    Confidence Stability:  {trust.confidence_stability:.2f}  (weight: 0.25)")

    print(f"\n  JUDGE'S FINAL ANSWER: {verdict.final_answer.upper()}")
    print(f"\n  EXPLANATION:")
    print(f"    {verdict.explanation}")
    print(f"\n  DEBATE SUMMARY:")
    print(f"    {verdict.debate_summary}")

    print(f"\n  TOKEN USAGE: {tokens['total_tokens']} tokens ({tokens['num_calls']} LLM calls)")

    print("\n" + "=" * 70)
    print("END OF ANALYSIS")
    print("=" * 70)


if __name__ == "__main__":
    main()
