import streamlit as st

from rag import save_uploaded_file, extract_text
from chunking import recursive_chunker
from embeddings import Embedder
from vector_store import VectorStore
from models import ChunkMetadata

st.set_page_config(page_title="Offline RAG")


@st.cache_resource
def get_embedder():
    return Embedder()


st.title("📄 Offline RAG Chatbot")

embedder = get_embedder()
vector_store = VectorStore()

st.sidebar.metric("Knowledge Base Size", vector_store.count())

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    st.write("✅ Embedding model loaded")

    file_path = save_uploaded_file(uploaded_file)
    st.success(f"Saved: {file_path.name}")

    text = extract_text(file_path)
    st.write("✅ Text extracted")

    chunks = recursive_chunker(text)
    st.write(f"✅ Created {len(chunks)} chunks")

    embeddings = embedder.embed(chunks)
    st.write("✅ Embeddings generated")

    metadata = []

    starting_chunk_id = vector_store.count()

    for index, chunk in enumerate(chunks):

        metadata.append(
            ChunkMetadata(
                chunk_id=starting_chunk_id + index,
                chunk_index=index,
                document_name=file_path.name,
                text=chunk
            )
        )

    vector_store.add(embeddings, metadata)
    vector_store.save()

    st.success(f"Knowledge Base now contains {vector_store.count()} chunks.")

    st.subheader("Chunk Information")

    st.write(f"Total Chunks: **{len(chunks)}**")

    for i, chunk in enumerate(chunks[:5]):
        st.text_area(
            label=f"Chunk {i + 1}",
            value=chunk,
            height=200,
            disabled=True
        )

    st.subheader("Embedding Information")

    st.write(f"Number of Embeddings: **{len(embeddings)}**")
    st.write(f"Embedding Dimension: **{len(embeddings[0])}**")

    st.write("First 10 values of first embedding:")
    st.write(embeddings[0][:10])