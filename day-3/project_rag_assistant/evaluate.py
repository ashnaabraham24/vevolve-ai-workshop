"""Small retrieval-and-answer smoke test for the RAG assistant."""

from ingest import PROJECT_DIR, ingest_directory
from ask import answer_question


TEST_CASES = [
    ("What is the first step in ingestion?", "load"),
    ("What metadata should chunks keep?", "source"),
]


def main():
    ingest_directory(PROJECT_DIR / "sample_docs")
    for question, expected_word in TEST_CASES:
        result = answer_question(question)
        passed = expected_word.lower() in result["answer"].lower()
        print(f"{'PASS' if passed else 'REVIEW'}: {question}")
        print(result["answer"])
        print()


if __name__ == "__main__":
    main()
