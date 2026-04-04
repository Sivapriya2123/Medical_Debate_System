from sentence_transformers import CrossEncoder


def load_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    Load reranker model.
    """
    model = CrossEncoder(model_name)
    return model


def rerank_results(reranker, query: str, candidates, top_k: int = 3):
    """
    Rerank retrieved candidates using a cross-encoder.
    """
    if not candidates:
        return []

    pairs = [(query, item["text"]) for item in candidates]
    scores = reranker.predict(pairs)

    reranked = []
    for item, score in zip(candidates, scores):
        item = dict(item)
        item["rerank_score"] = float(score)
        reranked.append(item)

    reranked = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)

    return reranked[:top_k]
