from rag.retriever import retrieve
from llm.local_llm import generate_response


def summarize_document():

    results = retrieve(
        "Summarize the entire document",
        k=10
    )

    context = "\n\n".join(
        results["documents"][0]
    )

    prompt = f"""
You are an expert research assistant.

Create a structured summary of this document.

Include:

1. Main Topics
2. Important Concepts
3. Key Formulas
4. Important Definitions
5. Final Takeaways

Context:

{context}

Summary:
"""

    return generate_response(prompt)