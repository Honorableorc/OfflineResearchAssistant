from rank_bm25 import BM25Okapi

bm25 = None
documents = []


def build_bm25(all_docs):

    global bm25
    global documents

    documents = all_docs

    tokenized_docs = [
        doc.split()
        for doc in documents
    ]

    bm25 = BM25Okapi(
        tokenized_docs
    )


def search_bm25(query, k=5):

    global bm25
    global documents

    if bm25 is None:
        return []

    scores = bm25.get_scores(
        query.split()
    )

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc
        for doc, score in ranked[:k]
    ]