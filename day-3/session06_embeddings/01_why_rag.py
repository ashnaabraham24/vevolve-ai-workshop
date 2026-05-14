"""Demo: Why RAG? Comparing Naive LLM vs RAG Approach.
Day 3, Session 6: Embeddings Part A
Learning Objective: Understand why RAG is needed vs naive LLM calls.
"""

import os

from common import get_openai_client, get_workshop_model


def naive_llm_call(query):
    """Ask LLM without context - prone to hallucinations."""
    client = get_openai_client()
    response = client.responses.create(
        model=get_workshop_model(), input=query, max_output_tokens=200
    )
    return response.output_text


def rag_call(query, context):
    """Ask LLM with relevant context - accurate answers."""
    client = get_openai_client()
    prompt = f"Context: {context}\n\nQuestion: {query}"
    response = client.responses.create(
        model=get_workshop_model(), input=prompt, max_output_tokens=200
    )
    return response.output_text


def main():
    query = "What topics are covered on Day 3 of the workshop?"

    # Naive approach - no context
    print("NAIVE LLM (no context):")
    print(naive_llm_call(query))
    print()

    # RAG approach - with context
    doc_path = os.path.join(os.path.dirname(__file__), "data", "sample_document.txt")
    with open(doc_path, "r") as f:
        context = f.read()

    print("RAG APPROACH (with context):")
    print(rag_call(query, context))


if __name__ == "__main__":
    main()
