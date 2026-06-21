from rag.vector_store import client
from llm.local_llm import generate_response


def generate_notes(filename):

    collection = client.get_collection(
        "documents"
    )

    results = collection.get(
        where={
            "filename": filename
        }
    )

    context = "\n\n".join(
        results["documents"]
    )

    prompt = f"""
You are an expert teacher.

Using ONLY the provided context, create detailed study notes.

Structure:

# Topic Name

## Definition

## Key Concepts

## Important Formulas

## Important Derivations

## Advantages

## Disadvantages

## Applications

## Exam Tips

## Key Takeaways

Context:

{context}

Notes:
"""

    return generate_response(prompt)