from typing import List, Dict


def recall_at_k(retrieved_ids: List[str], relevant_id: str, k: int) -> float:
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0


def hit_rate_at_k(retrieved_ids: List[str], relevant_id: str, k: int) -> float:
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0


def mrr_at_k(retrieved_ids: List[str], relevant_id: str, k: int) -> float:
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id == relevant_id:
            return 1.0 / rank
    return 0.0


def evaluate_queries(results_per_query: List[Dict], k: int = 5) -> Dict[str, float]:
    recalls = []
    hits = []
    mrrs = []

    for item in results_per_query:
        retrieved_ids = item["retrieved_ids"]
        relevant_id = item["relevant_id"]

        recalls.append(recall_at_k(retrieved_ids, relevant_id, k))
        hits.append(hit_rate_at_k(retrieved_ids, relevant_id, k))
        mrrs.append(mrr_at_k(retrieved_ids, relevant_id, k))

    return {
        f"Recall@{k}": sum(recalls) / len(recalls) if recalls else 0.0,
        f"HitRate@{k}": sum(hits) / len(hits) if hits else 0.0,
        f"MRR@{k}": sum(mrrs) / len(mrrs) if mrrs else 0.0,
    }
