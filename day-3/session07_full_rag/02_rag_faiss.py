"""Full RAG pipeline with FAISS vector DB instead of ChromaDB.
Day 3, Session 7: Full RAG Part A
Learning Objective: Implement end-to-end RAG workflow with FAISS.
"""

import os

import faiss
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
    print_separator("STAGE 2: Store embeddings in a FAISS index")

    embedding_matrix = np.array(doc_embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(dim)
    faiss.normalize_L2(embedding_matrix)
    index.add(embedding_matrix)
    print(f"  Created FAISS IndexFlatL2 with {dim} dimensions")
    print(f"  Stored {index.ntotal} vectors in the index")
    print("  ==> FAISS indexes vectors for fast similarity search (L2 distance)")

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
    print("  (FAISS computes L2 distance on normalized vectors = cosine similarity)\n")

    query_vec = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec, k=2)

    print(f"  Retrieved {len(indices[0])} most relevant documents:")
    for i, idx in enumerate(indices[0]):
        print(f"    {i + 1}. {docs[idx]}  (distance: {distances[0][i]:.4f})")
    print(
        "  ==> Relevant documents retrieved by vector similarity, not keyword matching"
    )

    # -- Stage 5: Generate answer ----------------------------------
    print_separator(
        "STAGE 5: Generate answer with LLM (Retrieval-Augmented Generation)"
    )

    contexts = [docs[idx] for idx in indices[0]]
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
