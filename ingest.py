import os
import uuid # unique ids for chunks
import hashlib # file hashing
import streamlit as st

from collections import Counter 
from sentence_transformers import SentenceTransformer # embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter # chunking

from db import add_chunks, get_all_chunks, file_hash_exists
from utils import extract_text_from_file


UPLOAD_DIR = "data/uploads"

# saves upload to directory
def save_uploaded_file(uploaded_file):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path

# initialize embedding model
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# splits file-text into chunks
def get_text_splitter(chunk_size=500, chunk_overlap=50):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

# computes unique file hash/fingerprint 
def compute_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

# process one uploaded file
def process_uploaded_file(uploaded_file, chunk_size=500, chunk_overlap=50):
    file_bytes = uploaded_file.getvalue()
    file_hash = compute_file_hash(file_bytes)

    # Duplicate check
    if file_hash_exists(file_hash):
        return {
            "success": False,
            "message": f"{uploaded_file.name} was already uploaded before.",
            "duplicate": True,
        }

    save_uploaded_file(uploaded_file)

    # Extract text from files
    extracted = extract_text_from_file(uploaded_file)
    if extracted is None:
        return {
            "success": False,
            "message": f"Unsupported file type: {uploaded_file.name}",
            "duplicate": False,
        }

    splitter = get_text_splitter(chunk_size, chunk_overlap)
    embedder = get_embedder()

    ids, documents, metadatas = [], [], []

    # PDF handling
    if extracted["type"] == "pdf":
        for page_data in extracted["content"]:
            chunks = splitter.split_text(page_data["text"])

            for chunk_index, chunk in enumerate(chunks):
                ids.append(str(uuid.uuid4()))
                documents.append(chunk)
                metadatas.append({
                    "document_name": uploaded_file.name,
                    "page": page_data["page"],
                    "chunk_index": chunk_index,
                    "file_hash": file_hash,
                })

    # TXT/DOCX handling
    else:
        chunks = splitter.split_text(extracted["content"])

        for chunk_index, chunk in enumerate(chunks):
            ids.append(str(uuid.uuid4()))
            documents.append(chunk)
            metadatas.append({
                "document_name": uploaded_file.name,
                "page": -1,
                "chunk_index": chunk_index,
                "file_hash": file_hash,
            })

    if not documents:
        return {
            "success": False,
            "message": f"No text found in {uploaded_file.name}",
            "duplicate": False,
        }

    # Create and store embeddings
    embeddings = embedder.encode(documents).tolist()
    add_chunks(ids, documents, metadatas, embeddings)

    return {
        "success": True,
        "message": f"{uploaded_file.name} processed successfully",
        "chunk_count": len(documents),
        "duplicate": False,
    }

# lists all uploads summarized
def summarize_documents():
    data = get_all_chunks()
    metadatas = data.get("metadatas", [])

    counter = Counter()
    for metadata in metadatas:
        doc_name = metadata.get("document_name", "unknown")
        counter[doc_name] += 1

    summary = [{"document_name": name, "chunk_count": count} for name, count in counter.items()]
    summary.sort(key=lambda x: x["document_name"].lower())
    return summary