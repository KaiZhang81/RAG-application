import chromadb
import streamlit as st


CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "rag_docs"

# initialize local ChromaDB and get collection 
@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection

# inserts chunks to DB
def add_chunks(ids, documents, metadatas, embeddings):
    collection = get_chroma_collection()
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

# chunk comparison and retrieval
def query_chunks(query_embedding, n_results=5):
    collection = get_chroma_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    return results

# retrieves all chunks from DB
def get_all_chunks():
    collection = get_chroma_collection()
    return collection.get()

# checks for duplicate file hash
def file_hash_exists(file_hash):
    collection = get_chroma_collection()
    result = collection.get(where={"file_hash": file_hash})
    ids = result.get("ids", [])
    return len(ids) > 0

# deletes docuemnts with matching chunks
def delete_document(document_name):
    collection = get_chroma_collection()
    data = collection.get(where={"document_name": document_name})
    ids = data.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)

# deletes all documents from DB
def delete_all_documents():
    collection = get_chroma_collection()
    data = collection.get()
    ids = data.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)