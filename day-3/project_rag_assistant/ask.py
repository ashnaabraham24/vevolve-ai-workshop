"""Retrieve relevant chunks and answer questions."""

import chromadb

from common import get_openai_client
from ingest import COLLECTION_NAME, DB_DIR

client = get_openai_client()


def embed_query(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    return response.data[0].embedding


def get_collection():
    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
    return chroma_client.get_collection(COLLECTION_NAME)


def answer_question(question: str, top_k: int = 3):

    collection = get_collection()

    # If no documents exist yet
    if collection.count() == 0:

        response = client.responses.create(
            model="gpt-5.4-nano",
            input=question,
            max_output_tokens=300,
        )

        return {
            "answer": response.output_text,
            "sources": [],
        }

    query_embedding = embed_query(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = "\n\n".join(documents)

    prompt = f"""
        You are a friendly and conversational AI assistant.
        Answer naturally like ChatGPT.
            DO NOT return:
            - JSON
            - structured reports
            - debugging formats
            - issue tracking formats
            If document context is relevant, use it.
            If document context is irrelevant or unavailable,
            answer using your general knowledge.

Document Context:
        Document Context:
        {context}

        Question:
        {question}
        """

    response = client.responses.create(
        model="gpt-5.4-nano",
        input=prompt,
        max_output_tokens=300,
    )

    return {
        "answer": response.output_text,
        "sources": [
            {
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "text": doc,
            }
            for doc, meta in zip(documents, metadatas)
        ],
    }