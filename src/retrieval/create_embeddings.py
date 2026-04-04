from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any


def load_embedding_model(
    model_name: str = "NeuML/pubmedbert-base-embeddings"
):
    """
    Biomedical embedding model for PubMed-style text.
    """
    model = SentenceTransformer(model_name)
    return model


def create_embeddings(model, documents: List[Dict[str, Any]]):
    texts = [doc["retrieval_text"] for doc in documents]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return embeddings
