from src.retrieval.load_dataset import load_pubmedqa
from src.retrieval.chunk_documents import chunk_documents
from src.retrieval.create_embeddings import load_embedding_model, create_embeddings
from src.retrieval.vector_store import create_chroma_collection, store_documents
from src.retrieval.bm25_index import build_bm25_index
from src.retrieval.retrieve_evidence import retrieve_evidence_hybrid


def build_retrieval_pipeline(
    limit: int = 500,
    collection_name: str = "pubmedqa_hybrid_collection",
    persist_directory: str = "./chroma_store",
):
    dataset = load_pubmedqa(limit=limit)
    documents = chunk_documents(dataset)

    model = load_embedding_model()
    embeddings = create_embeddings(model, documents)

    collection = create_chroma_collection(
        collection_name=collection_name,
        persist_directory=persist_directory
    )

    # avoid duplicate-add in repeated notebook runs
    if collection.count() == 0:
        store_documents(collection, documents, embeddings=embeddings)

    bm25 = build_bm25_index(documents)

    return collection, bm25, documents


def search_medical_evidence(query: str, top_k: int = 3, limit: int = 500):
    collection, bm25, documents = build_retrieval_pipeline(limit=limit)
    results = retrieve_evidence_hybrid(collection, bm25, documents, query, top_k=top_k)
    return results
