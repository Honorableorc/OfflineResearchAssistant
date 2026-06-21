import gradio as gr

from rag.rag_pipeline import answer_question
from rag.pdf_loader import load_pdf
from rag.chunker import create_chunks
from rag.embeddings import embed_chunks
from rag.notes_generator import generate_notes
from rag.mcq_generator import generate_mcqs

from rag.vector_store import (
    store_chunks,
    get_uploaded_documents,
    get_all_documents,
    reset_collection
)

from rag import document_store

from memory.sqlite import save_message

def clear_knowledge_base():

    reset_collection()

    return (
        "Knowledge Base Cleared",
        ""
    )

def refresh_documents():

    docs = get_uploaded_documents()

    if len(docs) == 0:
        return "No documents uploaded"

    return "\n".join(docs)


def upload_pdf(file):

    docs = load_pdf(file.name)

    document_store.documents = docs

    chunks = create_chunks(docs)

    filename = file.name.split("\\")[-1]

    for chunk in chunks:

        chunk["filename"] = filename

    embeddings = embed_chunks(chunks)

    store_chunks(chunks, embeddings)

    from rag.vector_store import get_all_documents
    from rag.bm25_store import build_bm25

    all_docs = get_all_documents()

    build_bm25(
    all_docs["documents"]
            )

    total_pages = len(docs)

    total_characters = sum(
        len(doc["text"])
        for doc in docs
    )

    status = (
        f"PDF Loaded Successfully\n\n"
        f"Pages: {total_pages}\n"
        f"Chunks: {len(chunks)}\n"
        f"Characters: {total_characters}"
    )

    documents = refresh_documents()
    pdf_list = get_uploaded_documents()

    return (
    status,
    documents,
    gr.update(
        choices=pdf_list
    ),
    gr.update(
        choices=pdf_list
    )
)


def create_notes(filename):

    return generate_notes(filename)


def create_mcqs(filename, topic, count):

    return generate_mcqs(
        filename,
        topic,
        int(count)
    )


def chat(message, history):

    save_message("user", message)

    response = answer_question(message)

    save_message("assistant", response)

    return response


with gr.Blocks(
    title="Offline Research Assistant"
) as demo:

    gr.Markdown(
        """
# 📚 Offline Research Assistant

### Local AI powered by Qwen2.5 + RAG + ChromaDB
"""
    )

    gr.Markdown("## 📄 Upload Documents")

    with gr.Row():

        pdf_file = gr.File(
            label="Upload PDF"
        )

        status_box = gr.Textbox(
            label="Upload Status"
        )

    gr.Markdown("## 📚 Uploaded Documents")

    documents_box = gr.Textbox(
        label="Knowledge Base",
        lines=5
    )
    clear_button = gr.Button(
    "🗑 Clear Knowledge Base"
)

    clear_button.click(
        clear_knowledge_base,
        outputs=[
        status_box,
        documents_box
        ]
    )

    

    gr.Markdown("## 📝 Research Notes Generator")
    notes_file = gr.Dropdown(
            choices=[],
            label="Select PDF"
        )
    notes_button = gr.Button(
            "📝 Generate Notes"
        )

    notes_output = gr.Textbox(
        label="Research Notes",
        lines=20
    )

    notes_button.click(
        create_notes,
        inputs=notes_file,
        outputs=notes_output
    )

    gr.Markdown("## 🎓 MCQ Generator")

    with gr.Row():
        mcq_file = gr.Dropdown(
            choices=[],
            label="Select PDF"
        )
        mcq_topic = gr.Textbox(
            label="Topic",
            placeholder="Maximum Likelihood Estimation"
        )

        mcq_count = gr.Slider(
            minimum=5,
            maximum=20,
            value=10,
            step=5,
            label="Number of Questions"
        )

    mcq_button = gr.Button(
        "🎓 Generate MCQs"
    )

    mcq_output = gr.Textbox(
        label="Generated MCQs",
        lines=25
    )

    mcq_button.click(
        create_mcqs,
        inputs=[
            mcq_file,
            mcq_topic,
            mcq_count
        ],
        outputs=mcq_output
    )
    pdf_file.upload(
        upload_pdf,
        inputs=pdf_file,
        outputs=[
            status_box,
            documents_box,
            notes_file,
            mcq_file
        ]
    )
    gr.Markdown("## 💬 Chat With Your Knowledge Base")

    gr.ChatInterface(
        fn=chat
    )