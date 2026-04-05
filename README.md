# Simple Local RAG Application

A local Retrieval-Augmented Generation (RAG) system built with Python and Streamlit that allows users to upload documents and ask questions about them using a locally running LLM.

---

## Features

- Upload PDF, DOCX, and TXT documents
- Automatic text extraction and chunking
- Semantic search using vector embeddings
- Local vector database (ChromaDB) for storage
- Question answering using local LLMs via Ollama
- Display of retrieved sources
- Document management:
  - View uploaded documents
  - Track number of chunks
  - Delete individual or all documents

---

## How It Works (RAG Pipeline)

    Upload document
        ↓
    Extract text
        ↓
    Split into chunks
        ↓
    Convert chunks → embeddings
        ↓
    Store in vector database (Chroma)
        ↓
    User asks question
        ↓
    Question → embedding
        ↓
    Find most similar chunks (vector search)
        ↓
    Send chunks + question → LLM (Ollama)
        ↓
    LLM generates answer

---

## Core Concepts

### Embeddings

Text is converted into numerical vectors using a pretrained model:

    SentenceTransformer("all-MiniLM-L6-v2")

These vectors represent the semantic meaning of the text.

---

### Vector Search (Similarity)

The system retrieves relevant information using cosine similarity:

    similarity(A, B) = (A · B) / (||A|| · ||B||)

- Similar meaning → vectors are close
- Different meaning → vectors are far apart

---

### Chunking

Documents are split into smaller parts:

- chunk_size: size of each text chunk (e.g., 500 characters)
- chunk_overlap: overlap between chunks (e.g., 50 characters)

This improves retrieval accuracy and preserves context.

---

### Top-K Retrieval

- top_k defines how many relevant chunks are retrieved
- Example: top_k = 3 → retrieve the 3 most relevant chunks

---

## Tech Stack

- Frontend: Streamlit  
- LLM: Ollama (local models such as llama3, qwen)  
- Embeddings: Sentence Transformers  
- Vector Database: ChromaDB (local, persistent)  
- Text Processing: PyMuPDF, python-docx  

---

## Local Data Storage

All data is stored locally:

    data/
    ├── uploads/        # original uploaded files
    └── chroma/         # vector database (embeddings + metadata)

ChromaDB uses:
- SQLite database (chroma.sqlite3)
- Binary vector storage

No external APIs or cloud services are required.

---

## Installation

### 1. Create virtual environment

    python -m venv .venv
    .venv\Scripts\activate

---

### 2. Install dependencies

    pip install streamlit chromadb sentence-transformers langchain-text-splitters pymupdf python-docx ollama

---

### 3. Install and run Ollama

Install Ollama from: https://ollama.com/

Pull a model:

    ollama pull llama3.2

---

### 4. Run the application

    streamlit run app.py

---

## Example Usage

1. Upload a document (PDF, DOCX, TXT)
2. Ask a question such as:
   - "What is the main topic of this document?"
3. The system:
   - retrieves relevant chunks
   - sends them to the LLM
   - generates an answer with context

---

## Why RAG

Traditional LLMs:
- No access to your documents
- Can produce incorrect or hallucinated answers

RAG systems:
- Ground responses in your data
- Improve accuracy
- Allow custom knowledge integration

---

## Possible Improvements

- Hybrid search (keyword + vector)
- Reranking retrieved chunks
- Caching frequently asked questions
- Multi-document comparison
- Evaluation metrics (precision and recall)
- Improved UI/UX

---

## Learning Takeaways

This project demonstrates:

- How LLMs can be combined with external knowledge
- How embeddings represent semantic meaning
- How vector databases enable efficient retrieval
- How to build a full-stack AI application locally

---

## Author

Kai Zhang
