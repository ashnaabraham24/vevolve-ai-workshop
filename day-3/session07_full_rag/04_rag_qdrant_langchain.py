"""Full RAG pipeline with LangChain + Qdrant (in-memory mode).
Day 3, Session 7: Full RAG Part A
Learning Objective: Implement end-to-end RAG workflow using LangChain abstractions.
"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
COLLECTION_NAME = "rag_docs_langchain"


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_chat_model(max_output_tokens: int = 200) -> ChatOpenAI:
    return ChatOpenAI(
        model=CHAT_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=max_output_tokens,
    )


def print_separator(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    qdrant_client = QdrantClient(":memory:")
    embeddings = get_embeddings()

    docs = [
        "Python is a programming language",
        "RAG combines LLM with retrieval",
        "Vector databases store embeddings for semantic search",
    ]
    documents = [
        Document(page_content=text, metadata={"source": f"doc-{i + 1}"})
        for i, text in enumerate(docs)
    ]

    # -- Stage 1: Embed documents ----------------------------------
    print_separator("STAGE 1: Convert documents into vector embeddings")

    print(f"  Input: {len(docs)} text documents")
    for d in docs:
        print(f"    - {d}")

    print(f"\n  Calling LangChain OpenAIEmbeddings with '{EMBEDDING_MODEL}'...")
    doc_embeddings = embeddings.embed_documents(docs)
    dim = len(doc_embeddings[0])
    print(
        f"  Output: {len(doc_embeddings)} vectors, each with {dim} dimensions"
    )
    print(f"  Preview of vector[0] (first 5 dims): {doc_embeddings[0][:5]}")
    print("  ==> LangChain wraps the embedding API behind an Embeddings interface")

    # -- Stage 2: Store vectors ------------------------------------
    print_separator("STAGE 2: Store embeddings in a Qdrant vector store")

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vector_store.add_documents(documents=documents)

    count = qdrant_client.count(collection_name=COLLECTION_NAME).count
    print(f"  Created Qdrant collection '{COLLECTION_NAME}' with COSINE distance")
    print(f"  Stored {count} LangChain Document objects")
    print("  ==> Qdrant stores vectors while LangChain manages documents + metadata")

    # -- Stage 3: Embed the query ----------------------------------
    print_separator("STAGE 3: Convert the user's question into an embedding")

    query = "What is RAG?"
    print(f'  Query: "{query}"')
    print(f"  Calling the SAME LangChain embedding model '{EMBEDDING_MODEL}'...")
    query_embedding = embeddings.embed_query(query)
    print(f"  Query vector preview (first 5 dims): {query_embedding[:5]}")
    print("  ==> Query is now in the same vector space as the documents")

    # -- Stage 4: Semantic search ----------------------------------
    print_separator("STAGE 4: Search for similar vectors (semantic search)")

    print("  Using LangChain similarity_search_with_score...")
    print("  (Qdrant computes cosine similarity internally)\n")

    hits = vector_store.similarity_search_with_score(query, k=2)

    print(f"  Retrieved {len(hits)} most relevant documents:")
    for i, (doc, score) in enumerate(hits, 1):
        print(f"    {i}. {doc.page_content}  (score: {score:.4f})")
    print(
        "  ==> LangChain returns Document objects from vector similarity search"
    )

    # -- Stage 5: Generate answer ----------------------------------
    print_separator(
        "STAGE 5: Generate answer with LLM (Retrieval-Augmented Generation)"
    )

    contexts = [doc.page_content for doc, _score in hits]
    context_text = "\n".join(f"- {ctx}" for ctx in contexts)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer using ONLY the provided context. If the context does not "
                "contain the answer, say: I don't have enough information to answer "
                "this question.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    chain = prompt | get_chat_model(max_output_tokens=200) | StrOutputParser()

    print("  Building a LangChain prompt -> chat model -> output parser chain")
    print(f"  Sending to chat model '{CHAT_MODEL}'...\n")
    answer = chain.invoke({"context": context_text, "question": query})
    print(f"  Final answer: {answer}")
    print("  ==> The LLM answers using only the retrieved context (grounding)")


if __name__ == "__main__":
    main()
