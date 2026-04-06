from sentence_transformers import SentenceTransformer
import ollama
import streamlit as st

from db import query_chunks

# initialize embedding model
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# retrieves top_k chunks as context
def retrieve_context(question, top_k=4):
    embedder = get_embedder()
    query_embedding = embedder.encode(question).tolist()

    # get most similiar chunks from DB
    results = query_chunks(query_embedding, n_results=top_k)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    sources = []
    for doc, metadata in zip(documents, metadatas):
        sources.append({
            "text": doc,
            "document_name": metadata.get("document_name", "unknown"),
            "page": metadata.get("page", -1),
            "chunk_index": metadata.get("chunk_index", -1),
        })

    return sources

# builds the prompt with context
def build_prompt(question, sources):
    context_blocks = []
    for i, source in enumerate(sources, start=1):
        page_info = f"page {source['page']}" if source["page"] != -1 else "no page"
        context_blocks.append(
            f"[Source {i} | {source['document_name']} | {page_info}]\n{source['text']}"
        )

    context = "\n\n".join(context_blocks)

    # actual prompt given to LLM
    prompt = f"""
            You are a helpful assistant for question answering over uploaded documents.
            Answer only based on the provided context.
            If the answer is not in the context, say that clearly.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """.strip()

    return prompt

# sends context and prompt to LLM
def ask_llm(question, model_name="llama3.1:latest", top_k=4):
    sources = retrieve_context(question, top_k=top_k)
    prompt = build_prompt(question, sources)

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response["message"]["content"]
    return answer, sources