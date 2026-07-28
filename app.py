import streamlit as st

from rag import save_uploaded_file, extract_text
from chunking import recursive_chunker
from embeddings import Embedder

st.set_page_config(page_title="Offline RAG")

st.title("📄 Offline RAG Chatbot")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:
    # Save the uploaded PDF
    file_path = save_uploaded_file(uploaded_file)

    st.success(f"Saved: {file_path.name}")

    st.write("✅ PDF uploaded")

    text = extract_text(file_path)
    st.write("✅ Text extracted")

    chunks = recursive_chunker(text)
    st.write(f"✅ Created {len(chunks)} chunks")

    embedder = Embedder()
    st.write("✅ Embedder created")

    embeddings = embedder.embed(chunks)
    st.write("✅ Embeddings generated")

    from vector_store import VectorStore

    # ...

    vector_store = VectorStore(embeddings.shape[1])

    vector_store.add(embeddings)

    st.success(f"Added {vector_store.count()} vectors to FAISS.")

    # -----------------------------
    # Chunk Information
    # -----------------------------
    st.subheader("Chunk Information")

    st.write(f"Total Chunks: **{len(chunks)}**")

    for i, chunk in enumerate(chunks[:5]):
        st.text_area(
            label=f"Chunk {i + 1}",
            value=chunk,
            height=200,
            disabled=True
        )

    # -----------------------------
    # Embedding Information
    # -----------------------------
    st.subheader("Embedding Information")

    st.write(f"Number of Embeddings: **{len(embeddings)}**")
    st.write(f"Embedding Dimension: **{len(embeddings[0])}**")

    st.write("First 10 values of the first embedding:")
    st.write(embeddings[0][:10])