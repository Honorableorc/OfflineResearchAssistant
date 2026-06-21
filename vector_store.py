from chromadb import PersistentClient

client = PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="documents"
)


def store_chunks(chunks, embeddings):

    ids = []

    documents = []

    metadatas = []

    for i, chunk in enumerate(chunks):

        ids.append(
            f"{chunk['filename']}_{chunk['chunk_id']}_{i}"
        )

        documents.append(chunk["text"])

        metadatas.append(
            {
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "filename": chunk["filename"]
            }
        )

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )


def get_uploaded_documents():

    data = collection.get()

    filenames = set()

    for meta in data["metadatas"]:

        filenames.add(
            meta["filename"]
        )

    return sorted(list(filenames))

def get_all_documents():

    return collection.get(
        include=["documents", "metadatas"]
    )

def reset_collection():

    global collection

    try:

        client.delete_collection(
            "documents"
        )

    except:

        pass

    collection = client.get_or_create_collection(
        name="documents"
    )