import chromadb
from typing import List, Dict, Any


def create_chroma_collection(
    collection_name: str = "pubmedqa_collection_v2",
    persist_directory: str = "./chroma_store_v2"
):
    client = chromadb.PersistentClient(path=persist_directory)

    collection = client.get_or_create_collection(name=collection_name)

    return collection


def store_documents(collection, documents: List[Dict[str, Any]], embeddings=None):
    ids = [doc["id"] for doc in documents]
    texts = [doc["retrieval_text"] for doc in documents]

    metadatas = []
    for doc in documents:
        metadatas.append(
            {
                "question": doc["question"][:500],
                "final_decision": doc["final_decision"][:100],
                "meshes": ", ".join(doc["meshes"][:10]) if doc["meshes"] else "",
                "labels": ", ".join(doc["labels"][:10]) if doc["labels"] else "",
                "year": str(doc["year"]) if doc["year"] else "",
            }
        )

    if embeddings is not None:
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings,
        )
    else:
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )
