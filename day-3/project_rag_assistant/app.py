"""Streamlit UI for the Day 3 RAG assistant."""

import shutil
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).parent))

from ask import answer_question
from ingest import PROJECT_DIR, ingest_directory

UPLOAD_DIR = PROJECT_DIR / "uploaded_docs"
UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="RAG Assistant")
st.title("RAG Knowledge Base Assistant")

uploaded_files = st.file_uploader(
    "Upload .txt or .pdf documents",
    type=["txt", "pdf"],
    accept_multiple_files=True,
)

col1, col2 = st.columns(2)
with col1:
    chunk_size = st.number_input(
        "Chunk size", min_value=200, max_value=1500, value=500, step=100
    )
with col2:
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=5, value=3)

if st.button("Ingest documents"):
    if uploaded_files:
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        UPLOAD_DIR.mkdir(exist_ok=True)
        for uploaded_file in uploaded_files:
            (UPLOAD_DIR / uploaded_file.name).write_bytes(uploaded_file.getbuffer())
        source_dir = UPLOAD_DIR
    else:
        source_dir = PROJECT_DIR / "sample_docs"

    with st.spinner("Ingesting documents..."):
        count = ingest_directory(source_dir, chunk_size=chunk_size)
    st.success(f"Ingested {count} chunks")

question = st.text_input("Ask a question", value="Why is metadata useful in RAG?")
if st.button("Ask"):
    with st.spinner("Retrieving and answering..."):
        result = answer_question(question, top_k=top_k)
    st.write(result["answer"])
    st.subheader("Retrieved Sources")
    for source in result["sources"]:
        st.markdown(f"**{source['source']}** chunk `{source['chunk_index']}`")
        st.caption(source["text"])
