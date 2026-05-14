"""Setup and query ChromaDB vector store.
Day 3, Session 6: Embeddings Part C
Learning Objective: Create a ChromaDB collection, add documents, query.
"""

import chromadb


def main():
    client = chromadb.Client()
    collection = client.create_collection("test")

    # Add documents
    collection.add(
        documents=["Doc1: Python is a language", "Doc2: Java is a language"],
        ids=["1", "2"],
    )

    # Query
    results = collection.query(query_texts=["What is Python?"], n_results=1)
    print("Query Results:", results)


if __name__ == "__main__":
    main()
