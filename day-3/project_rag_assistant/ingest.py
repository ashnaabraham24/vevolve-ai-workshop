"""Ingest text and PDF documents into a ChromaDB collection."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import chromadb
import pypdf

sys.path.append(str(Path(__file__).resolve().parents[2]))

from shared.common import embed_texts

PROJECT_DIR = Path(__file__).parent
DB_DIR = PROJECT_DIR / ".chroma"
COLLECTION_NAME = "workshop_rag"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, *, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_text_file(path: Path) -> list[tuple[str, dict]]:
    text = clean_text(path.read_text(encoding="utf-8"))
    return [(text, {"source": path.name, "page": "text"})]


def load_pdf_file(path: Path) -> list[tuple[str, dict]]:
    pages = []
    with path.open("rb") as file:
        reader = pypdf.PdfReader(file)
        for index, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                pages.append((text, {"source": path.name, "page": index}))
    return pages


def load_documents(directory: Path) -> list[tuple[str, dict]]:
    documents = []
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() == ".txt":
            documents.extend(load_text_file(path))
        elif path.suffix.lower() == ".pdf":
            documents.extend(load_pdf_file(path))

    print(f"Loaded {len(documents)} documents from {directory}")
    return documents


def get_collection(reset: bool = True):
    client = chromadb.PersistentClient(path=str(DB_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION_NAME)


def ingest_directory(
    directory: Path, *, chunk_size: int = 500, overlap: int = 80
) -> int:
    records = []
    for text, metadata in load_documents(directory):
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        print(f" {chunks}")
        for chunk_index, chunk in enumerate(chunks):
            records.append(
                {
                    "id": f"{metadata['source']}:{metadata['page']}:{chunk_index}",
                    "text": chunk,
                    "metadata": {**metadata, "chunk_index": chunk_index},
                }
            )

    if not records:
        raise ValueError(f"No .txt or .pdf documents found in {directory}")

    embeddings = embed_texts([record["text"] for record in records])
    collection = get_collection(reset=True)
    collection.add(
        ids=[record["id"] for record in records],
        documents=[record["text"] for record in records],
        metadatas=[record["metadata"] for record in records],
        embeddings=embeddings,
    )
    return len(records)


def main():
    count = ingest_directory(PROJECT_DIR / "sample_docs")
    print(f"Ingested {count} chunks into {DB_DIR}")


if __name__ == "__main__":
    main()
