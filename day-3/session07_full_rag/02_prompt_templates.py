"""Context-aware prompt templates for RAG.
Day 3, Session 7: Full RAG Part A
Learning Objective: Design prompts that use retrieved context.
"""

from common import ask_model


def basic_rag_prompt(context: str, query: str) -> str:
    """Basic RAG prompt - simple and effective."""
    return f"""Context: {context}

Question: {query}

Answer the question using ONLY the information provided in the context above.
If the context doesn't contain the answer, respond with "I don't have enough information to answer this question."
"""


def structured_rag_prompt(context: str, query: str) -> str:
    """Structured RAG prompt with clear delimiters."""
    return f"""### CONTEXT ###
{context}

### QUESTION ###
{query}

### INSTRUCTIONS ###
Answer the question using ONLY the information provided in the context section.
- Be concise and accurate
- Do not use external knowledge
- If the context is insufficient, say "I don't have enough information to answer this question."

### ANSWER ###
"""


def few_shot_rag_prompt(context: str, query: str) -> str:
    """Few-shot RAG prompt with examples of good answers."""
    return f"""Context: {context}

Question: {query}

Here are examples of good answers:
Example 1:
Context: "Python is a programming language known for its simplicity."
Question: "What is Python?"
Answer: Python is a programming language known for its simplicity.

Example 2:
Context: "RAG combines LLM with retrieval"
Question: "What is RAG?"
Answer: RAG is a technique that combines LLMs with retrieval systems.

Now answer the following:
Answer the question using ONLY the information provided in the context.
"""


def main():
    # Sample context and query
    context = "RAG reduces hallucinations by adding external context to the generation process."
    query = "Why use RAG?"

    print("=" * 60)
    print("BASIC RAG PROMPT")
    print("=" * 60)
    prompt1 = basic_rag_prompt(context, query)
    print(f"Prompt:\n{prompt1}\n")
    print(f"Response: {ask_model(prompt1, max_output_tokens=150)}\n")

    print("=" * 60)
    print("STRUCTURED RAG PROMPT")
    print("=" * 60)
    prompt2 = structured_rag_prompt(context, query)
    print(f"Prompt:\n{prompt2}\n")
    print(f"Response: {ask_model(prompt2, max_output_tokens=150)}\n")

    print("=" * 60)
    print("FEW-SHOT RAG PROMPT")
    print("=" * 60)
    prompt3 = few_shot_rag_prompt(context, query)
    print(f"Prompt:\n{prompt3}\n")
    print(f"Response: {ask_model(prompt3, max_output_tokens=150)}\n")


if __name__ == "__main__":
    main()
