"""Full RAG pipeline with Qdrant vector DB (in-memory mode).
Day 3, Session 7: Full RAG Part A
Learning Objective: Implement end-to-end RAG workflow with Qdrant.
"""

import os
import uuid

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

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
    qdrant = QdrantClient(":memory:")
    COLLECTION_NAME = "rag_docs"

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
    dim = len(doc_embeddings[0])
    print(
        f"  Output: {len(doc_embeddings)} vectors, each with {dim} dimensions"
    )
    print(f"  Preview of vector[0] (first 5 dims): {doc_embeddings[0][:5]}")
    print("  ==> Text is now represented as a dense vector for semantic comparison")

    # -- Stage 2: Store vectors ------------------------------------
    print_separator("STAGE 2: Store embeddings in a Qdrant collection")

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=i,
            vector=doc_embeddings[i],
            payload={"text": docs[i]},
        )
        for i in range(len(docs))
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    count = qdrant.count(collection_name=COLLECTION_NAME).count
    print(f"  Created Qdrant collection '{COLLECTION_NAME}' with COSINE distance")
    print(f"  Stored {count} points with text payloads")
    print("  ==> Qdrant indexes vectors and stores metadata in payloads")

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
    print("  (Qdrant computes cosine similarity internally)\n")

    search_result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=2,
    )
    hits = search_result.points

    print(f"  Retrieved {len(hits)} most relevant documents:")
    for i, hit in enumerate(hits, 1):
        text = hit.payload.get("text", "") if hit.payload else ""
        print(f"    {i}. {text}  (score: {hit.score:.4f})")
    print(
        "  ==> Relevant documents retrieved by vector similarity, not keyword matching"
    )

    # -- Stage 5: Generate answer ----------------------------------
    print_separator(
        "STAGE 5: Generate answer with LLM (Retrieval-Augmented Generation)"
    )

    contexts = [hit.payload.get("text", "") if hit.payload else "" for hit in hits]
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
