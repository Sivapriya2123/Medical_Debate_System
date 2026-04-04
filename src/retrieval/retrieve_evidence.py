from typing import List, Dict, Any


def dense_search(collection, model, query: str, top_k: int = 5):
    query_embedding = model.encode([query], normalize_embeddings=True)

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
    )

    retrieved_docs = results.get("documents", [[]])[0]
    retrieved_meta = results.get("metadatas", [[]])[0]
    retrieved_ids = results.get("ids", [[]])[0]
    retrieved_distances = (
        results.get("distances", [[]])[0]
        if "distances" in results
        else [0.0] * len(retrieved_ids)
    )

    output = []
    for doc_id, doc_text, meta, distance in zip(
        retrieved_ids, retrieved_docs, retrieved_meta, retrieved_distances
    ):
        output.append(
            {
                "id": doc_id,
                "text": doc_text,
                "metadata": meta,
                "score": float(distance),
                "source": "dense",
            }
        )

    return output


def reciprocal_rank_fusion(result_lists, k: int = 60):
    fused_scores = {}
    doc_store = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list, start=1):
            doc_id = item["id"]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (k + rank)
            doc_store[doc_id] = item

    reranked = sorted(
        doc_store.values(),
        key=lambda x: fused_scores[x["id"]],
        reverse=True
    )

    for item in reranked:
        item["hybrid_score"] = fused_scores[item["id"]]
        item["source"] = "hybrid"

    return reranked


def retrieve_evidence_hybrid(collection, model, bm25, documents, query: str, top_k: int = 3):
    from src.retrieval.bm25_index import bm25_search

    dense_results = dense_search(collection, model, query, top_k=8)
    sparse_results = bm25_search(bm25, documents, query, top_k=8)

    fused = reciprocal_rank_fusion([dense_results, sparse_results])

    return fused[:top_k]


def retrieve_evidence_hybrid_reranked(
    collection,
    model,
    bm25,
    documents,
    reranker,
    query: str,
    top_k: int = 3,
    candidate_k: int = 10,
):
    from src.retrieval.bm25_index import bm25_search
    from src.retrieval.reranker import rerank_results

    dense_results = dense_search(collection, model, query, top_k=candidate_k)
    sparse_results = bm25_search(bm25, documents, query, top_k=candidate_k)

    fused = reciprocal_rank_fusion([dense_results, sparse_results])
    fused_candidates = fused[:candidate_k]

    reranked = rerank_results(
        reranker=reranker,
        query=query,
        candidates=fused_candidates,
        top_k=top_k,
    )

    for item in reranked:
        item["source"] = "hybrid + reranker"

    return reranked
