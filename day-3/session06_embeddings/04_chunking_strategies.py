"""Compare different text chunking strategies.
Day 3, Session 6: Embeddings Part C
Learning Objective: Understand fixed-size, sentence-based, and overlap chunking.
"""


def fixed_chunk(text, size=100):
    return [text[i : i + size] for i in range(0, len(text), size)]


def sentence_chunk(text):
    import re

    return re.split(r"[.!?]+", text)


def overlap_chunk(text, size=100, overlap=20):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i : i + size])
    return chunks


def main():
    text = "Hello world. This is a test. Python is great. AI is the future."
    print("Fixed Chunks:", fixed_chunk(text))
    print("Sentence Chunks:", sentence_chunk(text))
    print("Overlap Chunks:", overlap_chunk(text))


if __name__ == "__main__":
    main()
