"""
FastAPI backend for the Medical Debate System.
Returns a single JSON response with all pipeline stages.

Run:
    cd Medical_Debate_System
    conda activate medical_debate
    uvicorn ui.backend.server:app --reload --port 8000
"""

import os
import re
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

app = FastAPI(title="Medical Debate System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy pipeline singleton ──────────────────────────────────────

_pipeline = {}


def _get_pipeline():
    if _pipeline:
        return _pipeline

    from src.retrieval.load_dataset import load_pubmedqa
    from src.retrieval.chunk_documents import chunk_documents
    from src.retrieval.create_embeddings import load_embedding_model, create_embeddings
    from src.retrieval.vector_store import create_chroma_collection, store_documents
    from src.retrieval.bm25_index import build_bm25_index
    from src.retrieval.reranker import load_reranker

    dataset = load_pubmedqa(limit=1000)
    documents = chunk_documents(dataset)
    model = load_embedding_model()
    embeddings = create_embeddings(model, documents)
    bm25 = build_bm25_index(documents)
    reranker = load_reranker()
    collection = create_chroma_collection(
        "pubmedqa_demo", os.path.join(PROJECT_ROOT, "chroma_store_demo")
    )
    if collection.count() == 0:
        store_documents(collection, documents, embeddings=embeddings)

    _pipeline.update({
        "collection": collection,
        "model": model,
        "bm25": bm25,
        "documents": documents,
        "reranker": reranker,
    })
    return _pipeline


# ── Helpers ──────────────────────────────────────────────────────

def _extract_citation_indices(text: str) -> list[int]:
    """Extract [1], [2], etc. from text and return 0-indexed list."""
    matches = re.findall(r"\[(\d+)\]", text)
    return list(set(int(m) - 1 for m in matches if int(m) >= 1))


# ── Request / Response ───────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str


# ── Main endpoint ────────────────────────────────────────────────

@app.post("/api/run")
async def run_pipeline(req: QuestionRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key or api_key == "your_openrouter_api_key_here":
        raise HTTPException(500, "OPENROUTER_API_KEY not set in .env")

    start_time = time.time()

    # ── 1. Load pipeline ──
    try:
        pipe = _get_pipeline()
    except Exception as e:
        raise HTTPException(500, f"Pipeline load failed: {e}")

    # ── 2. Retrieve evidence ──
    from src.retrieval.retrieve_evidence import retrieve_evidence_hybrid_reranked
    from src.retrieval.temporal_filter import filter_evidence_by_recency
    from src.retrieval.conflict_detector import add_conflict_metadata

    evidence = retrieve_evidence_hybrid_reranked(
        pipe["collection"], pipe["model"], pipe["bm25"],
        pipe["documents"], pipe["reranker"],
        query=question, top_k=5, candidate_k=15,
    )
    evidence = filter_evidence_by_recency(evidence, recency_threshold=2010, mode="flag")
    evidence, conflicts = add_conflict_metadata(evidence)

    retrieval_data = {
        "chunks": [],
        "total_candidates": 15,
        "after_reranking": len(evidence),
    }
    for i, e in enumerate(evidence):
        retrieval_data["chunks"].append({
            "index": i,
            "text": e.get("text", e.get("metadata", {}).get("context", "")),
            "source": e.get("id", e.get("metadata", {}).get("pmid", "")),
            "is_outdated": e.get("is_outdated", False),
            "has_conflict": e.get("has_conflict", False),
            "relevance_score": round(e.get("rerank_score", e.get("hybrid_score", 0)), 4),
        })

    # ── 3. Run debate ──
    from src.debate import run_debate
    from src.debate.agents import reset_token_usage, get_token_usage

    reset_token_usage()
    transcript = run_debate(question=question, evidence=evidence, max_rounds=2)

    # Organize by rounds
    rounds_data = {}
    for msg in transcript.messages:
        r = msg.round
        if r not in rounds_data:
            rounds_data[r] = {"round": r}
        key = "doctor_a" if msg.agent == "doctor_a" else "doctor_b"
        rounds_data[r][key] = {
            "position": msg.position,
            "confidence": msg.confidence,
            "reasoning": msg.reasoning,
            "evidence_cited": _extract_citation_indices(msg.reasoning),
            "full_response": msg.reasoning,
        }

    debate_data = {
        "rounds": [rounds_data[r] for r in sorted(rounds_data.keys())]
    }

    # ── 4. Trust + Judge ──
    from src.Judge.pipeline import run_judge_on_transcript

    verdict, trust, _ = run_judge_on_transcript(transcript=transcript)
    tokens = get_token_usage()

    trust_data = {
        "agreement_score": round(trust.agent_agreement, 2),
        "embedding_similarity": round(trust.reasoning_consistency, 2),
        "confidence_stability": round(trust.confidence_stability, 2),
        "composite_score": round(trust.overall, 3),
        "weights": {"agreement": 0.40, "similarity": 0.35, "stability": 0.25},
    }

    judge_data = {
        "prediction": verdict.final_answer,
        "reasoning": verdict.explanation,
        "full_response": verdict.debate_summary,
        "prompt_variant": "decisive_v1",
        "evidence_cited": _extract_citation_indices(verdict.explanation),
    }

    elapsed = round(time.time() - start_time, 1)

    return {
        "question": question,
        "retrieval": retrieval_data,
        "debate": debate_data,
        "trust": trust_data,
        "judge": judge_data,
        "metadata": {
            "total_time_seconds": elapsed,
            "model": "gpt-4o-mini",
            "debate_rounds": 2,
            "total_tokens": tokens["total_tokens"],
            "num_calls": tokens["num_calls"],
        },
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "pipeline_loaded": bool(_pipeline)}
