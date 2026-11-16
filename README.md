# Offline RAG Knowledge Base

A local, single-user Streamlit assistant for questions about text PDFs. PyMuPDF extracts pages; LangChain splits each page into 1,000-character chunks with 200-character overlap. BAAI/bge-small-en-v1.5 produces normalized 384-dimensional vectors. FAISS IndexFlatIP retrieves the nearest passages; local Ollama **qwen2.5:3b** generates an answer.

## Setup

Use Python 3.13 and the committed uv lockfile; do not install into global Python:

```sh
uv sync --locked
```

On this Mac, the restored CLI is `/Users/roshan/.local/share/ollama-runtime/ollama` (the old `/usr/local/bin/ollama` symlink remains broken). In each terminal, run `export PATH="$HOME/.local/share/ollama-runtime:$PATH"` before the Ollama commands below. Existing Qwen 3B model files were retained.

For a new installation, install Ollama separately. During an internet-connected, authorized setup, cache the models:

```sh
ollama pull qwen2.5:3b
.venv/bin/python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("BAAI/bge-small-en-v1.5")'
```

These setup commands may download model artifacts. The application loads BGE with `local_files_only=True`; it never pulls Ollama models. An incomplete local cache must be repaired during setup.

Start Ollama on loopback in one terminal, then Streamlit in another:

```sh
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NO_CLOUD=1 OLLAMA_NOPRUNE=1 ollama serve
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

For offline operation, complete setup first and disconnect external networking. The application uses the explicit loopback Ollama endpoint and disables environment proxies for that client. The Streamlit command disables usage statistics. Do not expose either service to the network. Fully disconnected end-to-end inference must still be checked on the target machine.

## Use

Attach a PDF using the bottom chat composer and submit, with or without a question. Selecting a file or changing settings alone does not ingest it. A question accompanying an invalid attachment is not answered. Duplicate PDFs reuse the existing index. Uploads are limited to 25 MB, 1,000 pages, and 2 million extracted characters. Empty, damaged, locked, and no-text PDFs are rejected; OCR is not included. Original page numbers are preserved when blank pages are skipped.

SHA-256 of the uploaded bytes identifies a document before extraction. Renaming identical bytes does not bypass duplicate detection; different PDFs with the same name remain separate. Original display names are metadata only. Original PDFs are retained as `uploads/<sha256>.pdf`. The composer resets after submission.

Ask a question of up to 2,000 characters and choose top-k. Up to 12,000 context characters are accepted; reduce top-k if this limit is exceeded. Answers stream progressively, with a 10-minute requested Ollama model keep-alive (bounded idle residency, not guaranteed under memory pressure), a 120-second network timeout, an 8,192-token requested context window, and up to 512 output tokens. Character limits are resource guards, not exact token accounting; unusual text can still exceed the effective model budget.

The sidebar prioritizes **Your chats**, with New Chat, conversation switching, and independent chat deletion. Knowledge-base management and retrieval settings are collapsed. Local model availability is cached for 30 seconds; it does not guarantee successful inference.

Native dual light/dark palettes live in `.streamlit/config.toml`. Choose appearance in the upper-right menu (System / Light / Dark); restart Streamlit after changing theme configuration. No custom runtime theme toggle is used. Viewer toolbar mode retains appearance; one semantic CSS selector hides the native recording entry (recheck after Streamlit upgrades). Retrieved passages open in a popover with an independently scrollable detail area, avoiding inline chat-height changes.

Session-only conversations survive reruns and switching. New Chat preserves previous chats without changing documents. Deleting a document removes conversations whose answers retrieved it, including their stored excerpts; unrelated chats remain. Conversations may disappear when the browser session ends or the server restarts. Interrupted streams are marked incomplete and are not saved as completed answers. Conversational retrieval uses up to three completed turns with retrieved passages (at most 3,900 history characters). Simple requests to explain the previous answer reuse its resolved retrieval query. Other common reference-based follow-ups are rewritten locally by Qwen into a standalone question before BGE retrieval. Clearly independent questions skip rewriting, including self-contained acronym definitions such as “What is FAISS and how is it useful?”. Unresolved or malformed rewrites prompt clarification; rewrite failures request a standalone question. Earlier assistant text is context, not factual evidence: answers must use freshly retrieved sources. The original question remains visible; the resolved retrieval query is available inside the source popover. Only the active chat contributes history. Retrieved passages show source IDs, filenames, full SHA-256 document IDs, original pages, document-relative chunk indices, derived chunk IDs (`document_id:chunk_index`), similarity scores, and text. Retrieval does not prove that a passage supports every claim; generated inline references are not automatically verified. Similarity scores are cosine similarities, not confidence probabilities.

Open **Knowledge base**, then a document’s popover, and confirm deletion. Deletion first publishes the revised index, then removes the matching stored PDF. A cleanup failure is reported; index deletion may already have succeeded. Matching legacy originals are removed only after verifying their content hash. Differently hashed legacy files are preserved. Failed ingestion removes newly created originals where possible; cleanup errors are reported. Files and index are not one filesystem transaction: abrupt termination between operations can leave an orphan PDF requiring manual hash-verified cleanup. No secure-erasure guarantee is made.

## Storage and recovery

Paths are anchored to the source directory, independent of the launch directory. The runtime `uploads/` and `knowledge_base/` directories are created only when saving.

`knowledge_base/snapshot.zip` contains `index.faiss` and `manifest.json`. The manifest includes a format version, embedding/preprocessing configuration, index SHA-256, and chunk metadata. Chunk identity is `(document_id, chunk_index)`; FAISS positions are internal ordering, not persistent IDs.

A complete temporary archive is flushed and fsynced, then published with one same-directory `os.replace`. Failed publication leaves the previous snapshot and in-memory state intact. ZIP CRCs, index checksum, configuration, index type/dimension, count, and metadata checks reject damaged/incompatible snapshots without resetting them. This is designed for trusted local storage; it does not authenticate maliciously modified files or make native FAISS deserialization safe for arbitrary downloads. Directory fsync and automatic power-loss recovery are not implemented; retain backups.

**Legacy compatibility:** old `faiss.index` / `metadata.json` pairs are preserved but not automatically imported. Their shared provenance and embedding configuration cannot be established from counts alone. If these files are detected without a new snapshot, startup stops with instructions. Back up both `knowledge_base/` and `uploads/`, move the old KB directory to a separate backup location, then re-ingest original PDFs through the application. Do not delete or rename originals blindly: filenames may have been overwritten by the old application. No automatic migration or deletion occurs.

For a damaged snapshot, stop the app, preserve the entire storage directory for inspection, and restore a trusted backup or move the damaged KB aside and re-ingest verified originals. Never edit checksums merely to bypass validation. If all storage is removed, the app has no external record with which to distinguish that from a new installation.

## Tests

```sh
.venv/bin/python -B -m unittest -v test_llm
```

Tests use real PyMuPDF, cached BGE and FAISS with synthetic PDFs and temporary storage. AppTest can submit native chat text, but its attachment driver is unavailable: attachment scenarios supply the real Streamlit `ChatInputValue` with synthetic file objects. This does not replace a browser attachment/appearance check. A network guard checks BGE loading and encoding without outbound connections. Ollama error responses are simulated; successful live generation requires a running local server/model and a separate manual check. Importing `test_llm` does not run tests or contact Ollama.

## Limitations

- Use **one browser session** for mutations. Local hosting does not prevent concurrent sessions; concurrent writes/processes are unsupported.
- No authentication, user isolation, encryption, or persistent conversation database. PDFs and extracted text remain plaintext local files.
- No OCR, table-layout reconstruction, document-scoped search, relevance threshold, reranking, or answer verification. English BGE embeddings have finite token limits; chunk defaults have not been tuned against a project benchmark.
- Prompts separate system instructions and mark passages as untrusted, but cannot eliminate hallucinations or prompt injection.
- Model revision pinning and multi-user storage are deferred. Keep backups and use only trusted local snapshots.
