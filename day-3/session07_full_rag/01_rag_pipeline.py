"""Full RAG pipeline: Query ==> Retrieve ==> Generate.
Day 3, Session 7: Full RAG Part A
Learning Objective: Implement end-to-end RAG workflow.
"""

import os

import chromadb
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")


def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def embed_texts(texts):
    client = get_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def ask_model(prompt: str, max_output_tokens: int = 200) -> str:
    client = get_client()
    response = client.responses.create(
        model=CHAT_MODEL,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )
    return response.output_text


def print_separator(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    chroma_client = chromadb.Client()

    try:
        chroma_client.delete_collection("rag_docs")
    except Exception:
        pass
    collection = chroma_client.create_collection("rag_docs")

    # -- Stage 1: Embed documents ----------------------------------
    print_separator("STAGE 1: Convert documents into vector embeddings")

    docs = [
        "Python is a programming language",
        "RAG combines LLM with retrieval",
        "Vector databases store embeddings for semantic search",
    ]
    print(f"  Input: {len(docs)} text documents")
    for d in docs:
        print(f"    - {d}")

    print(f"\n  Calling embedding model '{EMBEDDING_MODEL}'...")
    doc_embeddings = embed_texts(docs)
    print(
        f"  Output: {len(doc_embeddings)} vectors, each with {len(doc_embeddings[0])} dimensions"
    )
    print(f"  Preview of vector[0] (first 5 dims): {doc_embeddings[0][:5]}")
    print("  ==> Text is now represented as a dense vector for semantic comparison")

    # -- Stage 2: Store vectors ------------------------------------
    print_separator("STAGE 2: Store embeddings in a vector database")

    collection.add(
        documents=docs,
        embeddings=np.array(doc_embeddings, dtype=np.float32),
        ids=["1", "2", "3"],
    )
    print(f"  Stored {collection.count()} vectors in ChromaDB collection 'rag_docs'")
    print("  ==> ChromaDB indexes vectors for fast similarity search")

    # -- Stage 3: Embed the query ----------------------------------
    print_separator("STAGE 3: Convert the user's question into an embedding")

    query = "What is RAG?"
    print(f'  Query: "{query}"')
    print(f"  Calling the SAME embedding model '{EMBEDDING_MODEL}'...")
    query_embedding = embed_texts([query])[0]
    print(f"  Query vector preview (first 5 dims): {query_embedding[:5]}")
    print("  ==> Query is now in the same vector space as the documents")

    # -- Stage 4: Semantic search ----------------------------------
    print_separator("STAGE 4: Search for similar vectors (semantic search)")

    print("  Comparing query vector against all stored vectors...")
    print("  (ChromaDB computes cosine similarity internally)\n")
    results = collection.query(
        query_embeddings=np.array([query_embedding], dtype=np.float32),
        n_results=2,
    )

    docs_result = results.get("documents")
    if not docs_result or not docs_result[0]:
        print("  No relevant documents found!")
        return

    contexts = docs_result[0]
    dist_result = results.get("distances")
    distances = dist_result[0] if dist_result else None

    print(f"  Retrieved {len(contexts)} most relevant documents:")
    for i, ctx in enumerate(contexts, 1):
        score = f"  (distance: {distances[i - 1]:.4f})" if distances else ""
        print(f"    {i}. {ctx}{score}")
    print(
        "  ==> Relevant documents retrieved by vector similarity, not keyword matching"
    )

    # -- Stage 5: Generate answer ----------------------------------
    print_separator(
        "STAGE 5: Generate answer with LLM (Retrieval-Augmented Generation)"
    )

    context_text = "\n".join(f"- {ctx}" for ctx in contexts)
    prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using ONLY the information provided in the context above. "
        "If the context doesn't contain the answer, say 'I don't have enough information to answer this question.'"
    )

    print("  Building prompt with retrieved context + user question")
    print(f"  Sending to chat model '{CHAT_MODEL}'...\n")
    answer = ask_model(prompt, max_output_tokens=200)
    print(f"  Final answer: {answer}")
    print("  ==> The LLM answers using only the retrieved context (grounding)")


if __name__ == "__main__":
    main()
