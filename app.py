import streamlit as st

from rag import (
    save_uploaded_file,
    get_document_id,
    extract_pages
)

from chunking import recursive_chunker
from embeddings import Embedder
from vector_store import VectorStore
from models import ChunkMetadata
from llm import LLM


st.set_page_config(
    page_title="Offline RAG"
)


# ------------------------------------
# Load Models
# ------------------------------------

@st.cache_resource
def get_embedder():
    return Embedder()


@st.cache_resource
def get_llm():
    return LLM()


# ------------------------------------
# Initialize
# ------------------------------------

st.title("📄 Offline RAG Chatbot")

embedder = get_embedder()
llm = get_llm()
vector_store = VectorStore()


st.sidebar.metric(
    "Knowledge Base Size",
    vector_store.count()
)

# ------------------------------------
# Document Management
# ------------------------------------

st.sidebar.divider()

st.sidebar.header(
    "Documents"
)

documents = vector_store.get_documents()


if not documents:

    st.sidebar.write(
        "No documents uploaded yet."
    )


else:

    for document_id, document in documents.items():

        st.sidebar.write(
            f"**{document['document_name']}**"
        )

        st.sidebar.caption(
            f"{document['chunk_count']} chunks"
        )


        if st.sidebar.button(
            "Delete",
            key=document_id
        ):

            vector_store.delete_document(
                document_id
            )

            st.rerun()
            
# ------------------------------------
# Upload PDF
# ------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


if uploaded_file:

    # --------------------------------
    # Save File
    # --------------------------------

    file_path = save_uploaded_file(
        uploaded_file
    )

    st.success(
        f"Saved: {file_path.name}"
    )


    # --------------------------------
    # Generate Document ID
    # --------------------------------

    document_id = get_document_id(
        file_path
    )


    # --------------------------------
    # Check Duplicate
    # --------------------------------

    if vector_store.document_exists(
        document_id
    ):

        st.warning(
            "This document already exists "
            "in the knowledge base."
        )


    else:

        # --------------------------------
        # Extract Pages
        # --------------------------------

        pages = extract_pages(
            file_path
        )

        st.write(
            f"Text extracted from "
            f"{len(pages)} pages"
        )


        # --------------------------------
        # Create Chunks + Metadata
        # --------------------------------

        chunks = []
        metadata = []

        starting_chunk_id = vector_store.count()


        for page in pages:

            page_number = page[
                "page_number"
            ]

            page_text = page[
                "text"
            ]


            page_chunks = recursive_chunker(
                page_text
            )


            for chunk in page_chunks:

                chunk_id = (
                    starting_chunk_id
                    + len(metadata)
                )


                chunks.append(
                    chunk
                )


                metadata.append(

                    ChunkMetadata(

                        chunk_id=chunk_id,

                        document_id=document_id,

                        chunk_index=len(metadata),

                        document_name=file_path.name,

                        page_number=page_number,

                        text=chunk

                    )
                )


        st.write(
            f"Created {len(chunks)} chunks"
        )


        # --------------------------------
        # Generate Embeddings
        # --------------------------------

        with st.spinner(
            "Generating embeddings..."
        ):

            embeddings = (
                embedder.embed_documents(
                    chunks
                )
            )


        st.write(
            "Embeddings generated"
        )


        # --------------------------------
        # Add to Vector Store
        # --------------------------------

        vector_store.add(
            embeddings,
            metadata
        )


        # --------------------------------
        # Save Knowledge Base
        # --------------------------------

        vector_store.save()


        st.success(

            f"Knowledge Base now contains "
            f"{vector_store.count()} chunks."

        )


# ------------------------------------
# Ask Questions
# ------------------------------------

st.divider()

st.header("Ask Questions")


question = st.text_input(
    "Ask something about your documents"
)


top_k = st.slider(
    "Top K Results",
    min_value=1,
    max_value=10,
    value=5
)


# ------------------------------------
# Search + Generate Answer
# ------------------------------------

if st.button("Search"):

    if vector_store.count() == 0:

        st.error(
            "Knowledge Base is empty."
        )

        st.stop()


    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    # --------------------------------
    # Embed Query
    # --------------------------------

    query_embedding = (
        embedder.embed_query(
            question
        )
    )


    # --------------------------------
    # Search FAISS
    # --------------------------------

    results = vector_store.search(
        query_embedding,
        k=top_k
    )


    # --------------------------------
    # Build Context
    # --------------------------------

    context = "\n\n".join(

        result.metadata.text

        for result in results

    )


    # --------------------------------
    # Generate Answer
    # --------------------------------

    with st.spinner(
        "Generating answer..."
    ):

        answer = llm.generate_answer(
            question=question,
            context=context
        )


    # --------------------------------
    # Display Answer
    # --------------------------------

    st.subheader("Answer")

    st.write(
        answer
    )


    # --------------------------------
    # Display Sources
    # --------------------------------

    st.subheader("Sources")

    shown_sources = set()


    for result in results:

        source = (

            result.metadata.document_name,
            result.metadata.page_number

        )


        if source not in shown_sources:

            st.write(

                f"• {result.metadata.document_name} "
                f"— Page {result.metadata.page_number}"

            )

            shown_sources.add(
                source
            )


    # --------------------------------
    # Display Retrieved Chunks
    # --------------------------------

    st.subheader(
        "Retrieved Chunks"
    )


    for rank, result in enumerate(
        results,
        start=1
    ):

        with st.expander(

            f"Result {rank} | "
            f"Score: {result.score:.4f}"

        ):

            st.write(

                f"**Document:** "
                f"{result.metadata.document_name}"

            )

            st.write(

                f"**Page:** "
                f"{result.metadata.page_number}"

            )

            st.write(

                f"**Chunk ID:** "
                f"{result.metadata.chunk_id}"

            )

            st.write(

                f"**Chunk Index:** "
                f"{result.metadata.chunk_index}"

            )

            st.write(
                "**Text:**"
            )

            st.write(
                result.metadata.text
            )