import re
from typing import List, Dict, Any

from src.retrieval.temporal_filter import add_temporal_metadata


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def chunk_documents(dataset) -> List[Dict[str, Any]]:
    """
    Build retrieval documents with richer metadata.
    """
    documents = []

    for i, row in enumerate(dataset):
        pubid = str(row.get("pubid", i))
        question = clean_text(row.get("question", ""))

        context_obj = row.get("context", {})
        contexts = context_obj.get("contexts", [])
        labels = context_obj.get("labels", [])
        meshes = context_obj.get("meshes", [])

        combined_context = clean_text(" ".join(contexts))
        long_answer = clean_text(row.get("long_answer", ""))
        final_decision = clean_text(row.get("final_decision", ""))

        retrieval_text = clean_text(
            f"Question: {question} Context: {combined_context}"
        )

        documents.append(
            {
                "id": pubid,
                "question": question,
                "context": combined_context,
                "retrieval_text": retrieval_text,
                "labels": labels,
                "meshes": meshes,
                "long_answer": long_answer,
                "final_decision": final_decision,
                "year": None,   # keep placeholder for future temporal filtering
            }
        )

    # Enrich with temporal metadata (estimated year + recency flag)
    documents = add_temporal_metadata(documents, recency_threshold=2010)

    return documents
