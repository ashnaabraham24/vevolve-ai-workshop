"""Test RAG pipeline for hallucination failures.
Day 3, Session 7: Full RAG Part B
Learning Objective: Identify and test RAG failure modes.
"""

from common import ask_model


def main():
    # Test 1: Unanswerable query
    prompt = (
        "Context: No relevant context was retrieved.\n"
        "Question: What is the 2027 roadmap?\n"
        "Answer only from the context. If the answer is not present, say so."
    )
    print("Test 1 (Unanswerable):", ask_model(prompt, max_output_tokens=100))

    # Test 2: Conflicting context
    # Test 3: No context
    print("Run additional tests with conflicting/empty context")


if __name__ == "__main__":
    main()
