from rag.retriever import retrieve
from rag.bm25_store import search_bm25


def hybrid_retrieve(query):

    vector_results = retrieve(query)

    vector_docs = vector_results["documents"][0]

    vector_metadata = vector_results["metadatas"][0]

    bm25_docs = search_bm25(query)

    combined_docs = []

    combined_metadata = []

    for doc, meta in zip(
        vector_docs,
        vector_metadata
    ):

        if doc not in combined_docs:

            combined_docs.append(doc)

            combined_metadata.append(meta)

    return {
        "documents": combined_docs,
        "metadatas": combined_metadata
    }