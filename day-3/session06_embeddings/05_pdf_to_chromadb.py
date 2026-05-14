"""Full pipeline: PDF to ChromaDB.
Day 3, Session 6: Embeddings Part C
Learning Objective: Load PDF, chunk, embed, store in ChromaDB.
"""

from pathlib import Path

import chromadb
import pypdf
from common import get_openai_client


def main():
    # Load PDF
    pdf_path = Path(__file__).parent / "data" / "sample_document.pdf"
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        text = "".join(page.extract_text() for page in reader.pages)

    # Chunk text
    chunks = [text[i : i + 200] for i in range(0, len(text), 200)]

    # Store in ChromaDB
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("pdf_docs")
    collection.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])
    print("PDF loaded to ChromaDB. Chunks:", len(chunks))


if __name__ == "__main__":
    main()
