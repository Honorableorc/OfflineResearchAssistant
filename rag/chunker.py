from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []

    for doc in documents:

        page_number = doc["page"]

        texts = splitter.split_text(doc["text"])

        for chunk_id, text in enumerate(texts):

            chunks.append(
                {
                    "page": page_number,
                    "chunk_id": chunk_id,
                    "text": text
                }
            )

    return chunks
