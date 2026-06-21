from chromadb import PersistentClient
from rag.embeddings import embed_text


def retrieve(query, k=5):

    client = PersistentClient(path="chroma_db")

    collection = client.get_collection(
        name="documents"
    )

    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )

    return results