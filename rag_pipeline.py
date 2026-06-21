from rag.hybrid_retriever import hybrid_retrieve
from llm.local_llm import generate_response


def answer_question(question):

    results = hybrid_retrieve(question)

    context = "\n\n".join(
        results["documents"]
    )

    prompt = f"""
You are a research assistant.

Use the provided context to answer the question.

Explain concepts step-by-step whenever possible.

If equations are present, explain them.

If examples are present, use them.

Only say information is not found if the
context is clearly unrelated.

Context:

{context}

Question:

{question}

Answer:
"""

    response = generate_response(prompt)

    sources = []

    for meta in results["metadatas"]:

        source = (
            f"{meta['filename']} "
            f"(Page {meta['page']})"
        )

        sources.append(source)

    sources = sorted(set(sources))

    response += "\n\n📚 Sources:\n"

    for source in sources:

        response += f"\n• {source}"

    return response