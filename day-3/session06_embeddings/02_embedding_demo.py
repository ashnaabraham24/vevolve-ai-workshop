"""Generate and print text embeddings.
Day 3, Session 6: Embeddings Part B
Learning Objective: Understand what embeddings are and how to generate them.
"""

from common import get_openai_client


def main():
    client = get_openai_client()
    sentences = ["Hello world", "Goodbye world", "Python is great"]

    for sentence in sentences:
        response = client.embeddings.create(
            model="text-embedding-3-small", input=sentence
        )
        vector = response.data[0].embedding
        print(f"Sentence: {sentence}")
        print(f"Vector (first 5 dims): {vector[:5]}")
        print(f"Vector length: {len(vector)}\n")


if __name__ == "__main__":
    main()
