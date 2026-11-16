from pathlib import Path

import streamlit as st

from chunking import recursive_chunker
from embeddings import Embedder
from llm import LLM, recent_history
from models import ChunkMetadata
from rag import (MAX_PDF_BYTES, UPLOAD_FOLDER, delete_uploaded_file, extract_pages,
                 get_document_id, save_uploaded_file)
from vector_store import VectorStore

st.set_page_config(page_title="Offline RAG")


@st.cache_resource
def get_embedder():
    return Embedder()


@st.cache_resource
def get_llm():
    return LLM()


def new_chat():
    chat_id = max(st.session_state.chats, default=0) + 1
    st.session_state.chats[chat_id] = []
    st.session_state.active_chat = chat_id


def select_chat(chat_id):
    st.session_state.active_chat = chat_id


def delete_chat(chat_id):
    del st.session_state.chats[chat_id]
    if st.session_state.active_chat == chat_id:
        if st.session_state.chats:
            st.session_state.active_chat = next(reversed(st.session_state.chats))
        else:
            new_chat()


def delete_document(document_id, document_name):
    try:
        VectorStore().delete_document(document_id)
    except (OSError, ValueError, RuntimeError) as error:
        st.session_state.notice = ("error", f"Deletion was not saved: {error}")
        return
    affected = [chat_id for chat_id, messages in st.session_state.chats.items()
                if any(r.metadata.document_id == document_id
                       for m in messages for r in m.get("sources", []))]
    for chat_id in affected:
        delete_chat(chat_id)
    try:
        delete_uploaded_file(document_id, document_name)
    except (OSError, ValueError):
        st.session_state.notice = ("warning", "Index deleted; PDF cleanup failed. Inspect uploads.")
    else:
        st.session_state.notice = ("success", "Document deleted from the shared knowledge base. Affected chats removed.")


def add_pdf(uploaded_file, vector_store, embedder):
    retained_path = None
    created_original = False
    try:
        if not 0 < uploaded_file.size <= MAX_PDF_BYTES:
            raise ValueError("Select a nonempty PDF no larger than 25 MB.")
        data = uploaded_file.getvalue()
        document_id = get_document_id(data)
        if vector_store.document_exists(document_id):
            return "This PDF is already in the knowledge base; its existing passages are available."
        with st.spinner("Extracting, embedding, and saving PDF..."):
            pages = extract_pages(data)
            metadata = []
            for page in pages:
                for chunk in recursive_chunker(page["text"]):
                    if chunk.strip():
                        metadata.append(ChunkMetadata(
                            document_id=document_id, chunk_index=len(metadata),
                            document_name=Path(uploaded_file.name).name,
                            page_number=page["page_number"], text=chunk,
                        ))
            if not metadata:
                raise ValueError("This PDF produced no usable text chunks.")
            vectors = embedder.embed_documents([chunk.text for chunk in metadata])
            created_original = not (UPLOAD_FOLDER / f"{document_id}.pdf").exists()
            retained_path = save_uploaded_file(data, document_id)
            vector_store.add(vectors, metadata)
    except (OSError, ValueError, RuntimeError) as error:
        if created_original and retained_path is not None:
            try:
                retained_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                raise ValueError(f"{error} Original cleanup also failed; inspect uploads.") from cleanup_error
        raise
    return "Document added. You can now ask questions about it."


@st.cache_data(ttl=30)
def model_status():
    return get_llm().model_status()


def show_sources(results, retrieval_query=""):
    with st.popover(f"Retrieved passages · {len(results)}", icon=":material/description:"):
        st.caption("Retrieved context, not verification of every answer claim.")
        if retrieval_query:
            with st.expander("Retrieval query", type="compact"):
                st.text(retrieval_query)
        with st.container(height=380, border=False, autoscroll=False):
            for rank, result in enumerate(results, start=1):
                m = result.metadata
                with st.expander(f"S{rank} · {m.document_name} · Page {m.page_number}", type="compact"):
                    st.text(m.text)
                    with st.expander("Technical details", type="compact"):
                        st.text(f"Document: {m.document_name}\nPage: {m.page_number}\n"
                                f"Document ID: {m.document_id}\nChunk ID: {m.document_id}:{m.chunk_index}\n"
                                f"Chunk index: {m.chunk_index}\nSimilarity: {result.score:.4f}")


try:
    vector_store = VectorStore()
    embedder = get_embedder()
    llm = get_llm()
except (OSError, ValueError, RuntimeError) as error:
    st.error(str(error))
    st.stop()

st.session_state.setdefault("chats", {})
if not st.session_state.chats:
    new_chat()
if "notice" in st.session_state:
    level, message = st.session_state.pop("notice")
    getattr(st, level)(message)

with st.container(key="workspace-header", horizontal=True, vertical_alignment="center"):
    st.caption("Offline RAG", width="content")
    st.text(next((m["content"] for m in st.session_state.chats[st.session_state.active_chat]
                  if m["role"] == "user"), "New conversation")[:72], width="content")
with st.sidebar:
    st.button("New Chat", on_click=new_chat, width="stretch")
    st.caption("Your chats")
    with st.container(key="chat-navigation"):
        for chat_id, history in reversed(list(st.session_state.chats.items())):
            title = next((m["content"] for m in history if m["role"] == "user"), "New chat")
            with st.container(key="active-chat" if chat_id == st.session_state.active_chat else f"row-{chat_id}"):
                title_column, action_column = st.columns([6, 1], gap="small")
                title_column.button(title[:38], key=f"chat-{chat_id}", width="stretch", type="tertiary",
                                    on_click=select_chat, args=(chat_id,))
                with action_column.popover("⋯", help="Conversation actions"):
                    st.button("Delete chat", key=f"remove-{chat_id}", on_click=delete_chat, args=(chat_id,))
    with st.expander("Knowledge base"):
        st.caption(f"{len(vector_store.get_documents())} documents · {vector_store.count()} chunks")
        for document_id, document in vector_store.get_documents().items():
            with st.popover(document["document_name"], width="stretch"):
                st.write("Delete from the shared knowledge base? Chats using this document will be removed.")
                st.button("Confirm deletion", key=document_id, on_click=delete_document,
                          args=(document_id, document["document_name"]))
    with st.expander("Settings"):
        top_k = st.slider("Passages to retrieve", min_value=1, max_value=10, value=5)
        st.caption(f"{llm.model} · {model_status()}")
        st.caption("Status cached 30 seconds. Follow-ups use recent turns. Chats last only this session.")
        st.caption("Appearance: upper-right menu. Use one session for KB changes.")

messages = st.session_state.chats[st.session_state.active_chat]
st.html("""<style>
[data-testid="stMainMenuItem-recordScreencast"] { display: none; }
.st-key-workspace-header { padding-bottom: .6rem; border-bottom: 1px solid color-mix(in srgb, currentColor 9%, transparent); }
.st-key-workspace-header p { font-size: .8rem; opacity: .65; }
.st-key-chat-navigation [data-testid="stVerticalBlock"] { gap: .15rem; }
.st-key-chat-navigation button { min-height: 2.15rem; padding: .25rem .65rem; border: none; justify-content: flex-start; text-align: left; font-size: .875rem; }
.st-key-chat-navigation button[data-testid="stPopoverButton"] { padding: .25rem; background: transparent; opacity: .55; }
.st-key-chat-navigation [data-testid="stButton"] button > div { width: 100%; justify-content: flex-start; }
.st-key-chat-navigation button p { text-align: left; font-size: .84rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.st-key-active-chat { background: color-mix(in srgb, currentColor 6%, transparent); border-radius: .65rem; }
[data-testid="stSidebar"] summary { font-size: .8rem; opacity: .75; }
[data-testid="stSidebar"] [data-testid="stExpander"] > details { border: none; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { gap: .7rem; }
[data-testid="stChatMessage"] { background: transparent; padding: .6rem 0; line-height: 1.65; }
[data-testid="stChatMessage"] p { margin-bottom: .5rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse; margin-left: auto; width: fit-content; max-width: 88%; padding: .65rem 1rem;
    background: color-mix(in srgb, currentColor 6%, transparent); border-radius: 1.1rem; }
[data-testid="stChatMessageAvatarUser"] { display: none; }
[data-testid="stChatMessageAvatarAssistant"] { background: transparent; color: inherit; opacity: .65; width: 1.5rem; height: 1.5rem; }
[data-testid="stChatMessage"] button[data-testid="stPopoverButton"] {
    min-height: 1.8rem; padding: .2rem .65rem; border-radius: 2rem; font-size: .75rem;
    background: color-mix(in srgb, currentColor 3%, transparent); border-color: color-mix(in srgb, currentColor 10%, transparent); }
[data-testid="stChatMessage"] button[data-testid="stPopoverButton"] p { font-size: inherit; }
.st-key-welcome { padding-top: 16vh; text-align: center; }
.st-key-welcome h3 { font-size: 1.8rem; font-weight: 500; letter-spacing: -.045rem; text-align: center; }
.st-key-welcome p { text-align: center; opacity: .65; }
button { border-radius: .65rem; transition: background-color 180ms ease, border-color 180ms ease; }
button:active { filter: brightness(.94); }
[data-testid="stChatInput"] {
    overflow: hidden; border-radius: 1.35rem; border: 1px solid color-mix(in srgb, currentColor 13%, transparent);
    box-shadow: 0 4px 24px #0000000a; transition: border-color 180ms ease, box-shadow 180ms ease; }
[data-testid="stChatInput"]:focus-within { border-color: color-mix(in srgb, currentColor 28%, transparent); box-shadow: 0 5px 26px #00000012; }
[data-testid="stChatInput"] > div { border: none; box-shadow: none; }
[data-testid="stChatInput"] textarea:focus { outline: none; box-shadow: none; }
[data-testid="stChatInput"] button { color: inherit; }
@media (prefers-reduced-motion: no-preference) {
    .st-key-incoming { animation: appear 180ms ease-out; }
    @keyframes appear { from { opacity: 0; } to { opacity: 1; } }
}
@media (prefers-reduced-motion: reduce) { button, [data-testid="stChatInput"] { transition: none; } }
</style>""")
if not messages:
    with st.container(key="welcome"):
        st.markdown("◇")
        st.subheader("What would you like to explore?")
        st.caption("Attach a text PDF below, then ask a question." if not vector_store.count()
                   else "Ask about findings, methods, or limitations in your documents.")
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"], message.get("retrieval_query", ""))

submission = st.chat_input("Message or attach a PDF", accept_file=True, file_type=["pdf"],
                           max_upload_size=25, max_chars=2000)
if submission and (submission.text.strip() or submission.files):
    history = recent_history(messages)
    question = submission.text.strip()
    uploaded_file = submission.files[0] if submission.files else None
    messages.append({"role": "user", "content": question or f"Upload {uploaded_file.name}"})
    with st.container(key="incoming"), st.chat_message("user"):
        st.markdown(messages[-1]["content"])
    attachment_ok = True
    if uploaded_file is not None:
        try:
            confirmation = add_pdf(uploaded_file, vector_store, embedder)
        except (OSError, ValueError, RuntimeError) as error:
            attachment_ok = False
            confirmation = f"PDF was not added: {error}"
        messages.append({"role": "assistant", "content": confirmation})
        with st.chat_message("assistant"):
            st.markdown(confirmation)
    if question and attachment_ok:
        messages.append({"role": "assistant", "content": "Answer interrupted. Send the question again to retry."})
        with st.chat_message("assistant"):
            output = st.empty()
            try:
                if not vector_store.count():
                    raise ValueError("Attach a text PDF before asking document questions.")
                with st.spinner("Finding relevant passages…"):
                    retrieval_query = llm.resolve_question(question, history)
                    if retrieval_query is None:
                        messages[-1] = {"role": "assistant", "content": "Which subject do you mean? Please name it so I can search the documents."}
                        st.rerun()
                    results = vector_store.search(embedder.embed_query(retrieval_query), k=top_k)
                    context = "\n\n".join(
                        f"[S{rank}] {r.metadata.document_name}, page {r.metadata.page_number}\n{r.metadata.text}"
                        for rank, r in enumerate(results, start=1)
                    )
                with output.container(), st.spinner("Writing an answer…"):
                    answer = st.write_stream(llm.stream_answer(question, context, history))
                if not answer:
                    raise ValueError("Ollama returned an empty answer. Please retry.")
            except (OSError, ValueError, RuntimeError) as error:
                output.empty()
                answer = f"Answer not completed: {error} Please send the question again to retry."
                st.warning(answer)
                results = []
            messages[-1] = {"role": "assistant", "content": answer, "sources": results,
                            "retrieval_query": retrieval_query if results else ""}
            if results:
                show_sources(results, retrieval_query)
    st.rerun()
