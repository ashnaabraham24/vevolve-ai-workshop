"""RAG query demo: answer with retrieved source chunks."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "project_rag_assistant"))

from ask import answer_question
from ingest import ingest_directory


def main():
    project_dir = Path(__file__).resolve().parents[1] / "project_rag_assistant"
    ingest_directory(project_dir / "sample_docs")
    result = answer_question("What should happen before creating embeddings?")
    print("Answer:")
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source['source']} chunk {source['chunk_index']}")


if __name__ == "__main__":
    main()
