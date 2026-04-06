import streamlit as st

from ingest import process_uploaded_file, summarize_documents
from rag import ask_llm
from db import delete_document, delete_all_documents

# page title
st.set_page_config(page_title="Local RAG Application", layout="wide")
st.title("Local RAG Application")

# page sidebar configurations
with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("Ollama model", value="llama3.1:latest")
    top_k = st.slider("Top-K chunks", min_value=1, max_value=10, value=4)
    chunk_size = st.slider("Chunk size", min_value=200, max_value=1200, value=500, step=50)
    chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=300, value=50, step=10)

tab1, tab2, tab3 = st.tabs(["Ask", "Upload", "Manage Documents"])

# tab1: LLM-Chatbot
with tab1:
    st.subheader("Ask a question")

    # User input for query
    question = st.text_area("Your question", height=120)

    if st.button("Get answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                # Run RAG pipeline (retrieve + generate)
                answer, sources = ask_llm(question, model_name=model_name, top_k=top_k)

                # Show generated answer
                st.markdown("### Answer")
                st.write(answer)

                # Show retrieved context chunks
                st.markdown("### Retrieved sources")
                for i, source in enumerate(sources, start=1):
                    page_label = source["page"] if source["page"] != -1 else "N/A"

                    # Expandable source view
                    with st.expander(f"Source {i}: {source['document_name']} | page: {page_label}"):
                        st.write(source["text"])

            except Exception as e:
                st.error(f"Error while generating answer: {e}")


# tab2: document uploads
with tab2:
    st.subheader("Upload documents")

    # Upload files
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if st.button("Process uploaded files"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            for uploaded_file in uploaded_files:
                try:
                    # Process file (hash → extract → chunk → embed → store)
                    result = process_uploaded_file(
                        uploaded_file,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )

                    # Handle result states
                    if result["success"]:
                        st.success(f"{uploaded_file.name}: {result['chunk_count']} chunks created")
                    elif result.get("duplicate", False):
                        st.warning(result["message"])
                    else:
                        st.error(result["message"])

                except Exception as e:
                    st.error(f"{uploaded_file.name}: {e}")


# tab3: manage documents
with tab3:
    st.subheader("Stored documents")

    # Get document overview from DB
    docs = summarize_documents()

    if not docs:
        st.info("No documents stored yet.")
    else:
        for doc in docs:
            col1, col2 = st.columns([4, 1])

            # Show document name and chunk count
            with col1:
                st.write(f"**{doc['document_name']}** — {doc['chunk_count']} chunks")

            # Delete single document
            with col2:
                if st.button("Delete", key=f"delete_{doc['document_name']}"):
                    deleted = delete_document(doc["document_name"])
                    st.success(f"Deleted {deleted} chunks from {doc['document_name']}")
                    st.rerun()

        st.markdown("---")

        # Delete all documents at once
        if st.button("Delete ALL documents"):
            deleted = delete_all_documents()
            st.warning(f"Deleted {deleted} chunks in total.")
            st.rerun()