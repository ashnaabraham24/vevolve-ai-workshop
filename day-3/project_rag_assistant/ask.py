"""Ask questions against the Day 3 RAG knowledge base."""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ingest import COLLECTION_NAME, DB_DIR

from shared.common import ask_model, embed_texts


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def retrieve(question: str, *, top_k: int = 3) -> list[dict]:
    question_embedding = embed_texts([question])[0]
    results = get_collection().query(
        query_embeddings=[question_embedding],
        n_results=top_k,
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    return [{"text": text, **metadata} for text, metadata in zip(documents, metadatas)]


def answer_question(question: str, *, top_k: int = 3) -> dict:
    chunks = retrieve(question, top_k=top_k)
    print(chunks)
    context = "\n\n".join(
        f"Source: {chunk['source']} chunk {chunk['chunk_index']}\n{chunk['text']}"
        for chunk in chunks
    )
    prompt = f"""
    Answer the question using only the context below.
    If the context is not enough, say: I do not have enough information.
    Include a short "Sources" line at the end.

    Context:
    {context}

    Question:
    {question}
    """
    return {"answer": ask_model(prompt, max_output_tokens=600), "sources": chunks}


def main():
    question = "Why is metadata useful in a RAG system?"
    result = answer_question(question)
    print(result["answer"])
    print("\nRetrieved chunks:")
    for source in result["sources"]:
        print(f"- {source['source']} chunk {source['chunk_index']}")


if __name__ == "__main__":
    main()
