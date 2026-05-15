"""Streamlit chatbot + RAG assistant."""

import shutil

import streamlit as st
from dotenv import load_dotenv

from ask import answer_question
from ingest import PROJECT_DIR, ingest_directory

# Load environment variables
load_dotenv()

# Upload directory
UPLOAD_DIR = PROJECT_DIR / "uploaded_docs"
UPLOAD_DIR.mkdir(exist_ok=True)

# Page settings
st.set_page_config(page_title="AI Chatbot")

st.title("AI Chatbot ")

with st.sidebar:

    st.header("Settings")

    chunk_size = st.number_input(
        "Chunk Size",
        min_value=200,
        max_value=1500,
        value=500,
        step=100,
    )

    top_k = st.slider(
        "Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=3,
    )



uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)


if st.button("Ingest Documents"):

    if uploaded_files:

        # Clear old uploads
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        UPLOAD_DIR.mkdir(exist_ok=True)

        # Save uploaded files
        for uploaded_file in uploaded_files:
            file_path = UPLOAD_DIR / uploaded_file.name
            file_path.write_bytes(uploaded_file.getbuffer())

        # Ingest documents
        with st.spinner("Ingesting documents..."):

            count = ingest_directory(
                UPLOAD_DIR,
                chunk_size=chunk_size,
            )

        st.success(f"Ingested {count} chunks!")

    else:
        st.warning("Please upload at least one PDF or TXT file.")



if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Ask something about your documents...")

if user_input:

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # Generate response
    with st.spinner("Thinking..."):

        try:

            result = answer_question(
                user_input,
                top_k=top_k,
            )

        except Exception:

            result = {
                "answer": "Please upload and ingest documents first, or ask a normal question.",
                "sources": [],
            }

    assistant_response = result["answer"]

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_response,
        }
    )

    with st.expander("Retrieved Sources"):

        for source in result["sources"]:

            st.markdown(
                f"**{source['source']}** "
                f"(chunk {source['chunk_index']})"
            )

            st.info(source["text"])