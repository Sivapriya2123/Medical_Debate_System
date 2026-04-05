"""
End-to-end test: Retrieval Pipeline → Debate System
Run: python main.py
"""

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.load_dataset import load_pubmedqa
from src.retrieval.chunk_documents import chunk_documents
from src.retrieval.create_embeddings import load_embedding_model, create_embeddings
from src.retrieval.vector_store import create_chroma_collection, store_documents
from src.retrieval.bm25_index import build_bm25_index
from src.retrieval.reranker import load_reranker
from src.retrieval.retrieve_evidence import retrieve_evidence_hybrid_reranked
from src.debate import run_debate
from src.Judge.pipeline import run_judge_on_transcript
from src.Judge.evaluation import (
    compute_evaluation_report,
    export_results_to_csv,
    export_results_to_json,
    print_comparison_table,
)



def main():
    # ── 1. Build retrieval pipeline ──
    print("Loading PubMedQA dataset...")
    dataset = load_pubmedqa(limit=200)
    documents = chunk_documents(dataset)
    print(f"Loaded {len(documents)} documents")

    print("Creating embeddings...")
    model = load_embedding_model()
    embeddings = create_embeddings(model, documents)

    print("Building BM25 index and reranker...")
    bm25 = build_bm25_index(documents)
    reranker = load_reranker()

    collection = create_chroma_collection("pubmedqa_test", "./chroma_store_test")
    if collection.count() == 0:
        store_documents(collection, documents, embeddings=embeddings)
    print(f"ChromaDB ready: {collection.count()} docs")

    # ── 2. Pick a question and retrieve evidence ──
    question = documents[0]["question"]
    ground_truth = documents[0]["final_decision"]

    print(f"\nQuestion: {question}")
    print(f"Ground truth: {ground_truth}")

    print("\nRetrieving evidence...")
    evidence = retrieve_evidence_hybrid_reranked(
        collection, model, bm25, documents, reranker,
        query=question, top_k=3, candidate_k=10,
    )
    for i, item in enumerate(evidence, 1):
        print(f"  Evidence {i}: {item['metadata']['question'][:80]}...")

    # ── 3. Run debate ──
    print("\nRunning debate (2 rounds)...")
    transcript = run_debate(question=question, evidence=evidence, max_rounds=2)

    # ── 4. Print results ──
    print("\n" + "=" * 60)
    print("DEBATE TRANSCRIPT")
    print("=" * 60)

    for msg in transcript.messages:
        print(f"\n[Round {msg.round}] {msg.agent.upper()}")
        print(f"  Position:   {msg.position}")
        print(f"  Confidence: {msg.confidence}")
        print(f"  Evidence:   {msg.evidence_cited}")
        print(f"  Reasoning:  {msg.reasoning[:300]}...")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Doctor A: {transcript.doctor_a_final_position} (confidence: {transcript.doctor_a_final_confidence})")
    print(f"Doctor B: {transcript.doctor_b_final_position} (confidence: {transcript.doctor_b_final_confidence})")
    print(f"Ground truth: {ground_truth}")
    print("=" * 60)

    # ── 5. Run judge pipeline (reuses the transcript from step 3) ──
    print("\nRunning judge pipeline...")
    verdict, trust, result = run_judge_on_transcript(
        transcript=transcript,
        ground_truth=ground_truth,
    )

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
    if result:
        print(f"\nCorrect: {result.correct}")
    print("=" * 60)

    # ── 6. Save artifacts to outputs/ ──
    if result:
        report = compute_evaluation_report([result], system_name="full_system")
        reports = [report]

        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        print_comparison_table(reports)

        csv_path = export_results_to_csv(reports)
        json_path = export_results_to_json(reports)
        print(f"\nArtifacts saved:")
        print(f"  CSV  : {csv_path}")
        print(f"  JSON : {json_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
