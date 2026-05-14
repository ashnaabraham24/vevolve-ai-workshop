"""Basic RAG evaluation demo using expected source checks."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "project_rag_assistant"))

from ask import answer_question
from ingest import ingest_directory


TEST_CASES = [
    {
        "question": "What is the first step in the ingestion pipeline?",
        "expected_source": "rag_notes.txt",
        "must_contain": "load",
    },
    {
        "question": "Why do we keep metadata with chunks?",
        "expected_source": "rag_notes.txt",
        "must_contain": "source",
    },
]


def main():
    project_dir = Path(__file__).resolve().parents[1] / "project_rag_assistant"
    ingest_directory(project_dir / "sample_docs")

    for case in TEST_CASES:
        result = answer_question(case["question"])
        sources = {source["source"] for source in result["sources"]}
        source_ok = case["expected_source"] in sources
        answer_ok = case["must_contain"].lower() in result["answer"].lower()
        status = "PASS" if source_ok and answer_ok else "REVIEW"
        print(f"{status}: {case['question']}")
        print(f"  Sources: {sorted(sources)}")
        print(f"  Answer: {result['answer'][:180]}")


if __name__ == "__main__":
    main()
