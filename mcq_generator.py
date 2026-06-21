from rag.vector_store import client
from llm.local_llm import generate_response


def generate_mcqs(
    filename,
    topic,
    num_questions=10
):

    collection = client.get_collection(
        "documents"
    )

    results = collection.get(
        where={
            "filename": filename
        },
        limit=5
    )

    context = "\n\n".join(
        results["documents"]
    )

    prompt = f"""
You are an expert professor.

Using ONLY the provided context:

Generate {num_questions} multiple-choice questions.

For EACH question provide:

Question

A) Option

B) Option

C) Option

D) Option

Correct Answer: <Option Letter>

Explanation: <1-2 line explanation>

Rules:

- Use only the context.
- Cover different concepts.
- Make questions exam-oriented.
- Keep explanations short.
- Format neatly.

Topic:

{topic}

Context:

{context}

MCQs:
"""

    return generate_response(prompt)