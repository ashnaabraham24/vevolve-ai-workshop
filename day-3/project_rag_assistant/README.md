# Project: RAG Knowledge Base Assistant

This project shows the two halves of RAG:

1. `ingest.py` loads documents, cleans text, chunks it, embeds chunks, and stores
   them in ChromaDB.
2. `ask.py` retrieves relevant chunks and asks the model to answer with sources.

Run:

```bash
python day-3/project_rag_assistant/ingest.py
python day-3/project_rag_assistant/ask.py
python day-3/project_rag_assistant/evaluate.py
```

Optional UI:

```bash
streamlit run day-3/project_rag_assistant/app.py
```
