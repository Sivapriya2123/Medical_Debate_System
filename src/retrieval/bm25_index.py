from rank_bm25 import BM25Okapi


def tokenize(text: str):
    return text.lower().split()


def build_bm25_index(documents):
    corpus = [tokenize(doc["retrieval_text"]) for doc in documents]
    bm25 = BM25Okapi(corpus)
    return bm25


def bm25_search(bm25, documents, query: str, top_k: int = 5):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    output = []
    for doc, score in ranked:
        output.append(
            {
                "id": doc["id"],
                "text": doc["retrieval_text"],
                "metadata": {
                    "question": doc["question"],
                    "final_decision": doc["final_decision"],
                    "meshes": ", ".join(doc["meshes"][:10]) if doc["meshes"] else "",
                    "labels": ", ".join(doc["labels"][:10]) if doc["labels"] else "",
                },
                "score": float(score),
                "source": "bm25",
            }
        )
    return output
