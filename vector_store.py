import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from embeddings import EMBEDDING_CONFIG
from models import ChunkMetadata, SearchResult

INDEX_DIR = Path(__file__).resolve().parent / "knowledge_base"


class VectorStore:
    def __init__(self, directory=INDEX_DIR):
        self.path = Path(directory) / "snapshot.zip"
        self.index = faiss.IndexFlatIP(EMBEDDING_CONFIG["dimension"])
        self.metadata = []
        if self.path.exists():
            self.load()
        elif any((self.path.parent / name).exists() for name in ("faiss.index", "metadata.json")):
            raise ValueError(
                "Legacy knowledge base preserved unchanged. Its index and metadata cannot be verified "
                "as one snapshot. Back up knowledge_base and uploads, move the legacy KB aside, "
                "then re-ingest the original PDFs. See README."
            )

    def add(self, embeddings, metadata):
        vectors = np.asarray(embeddings, dtype="float32")
        if (vectors.ndim != 2 or not len(vectors)
                or vectors.shape != (len(metadata), self.index.d)
                or not np.isfinite(vectors).all()):
            raise ValueError("Embeddings must be finite, nonempty rows matching metadata and dimension.")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
            raise ValueError("Embeddings must be normalized.")
        identities = {(m.document_id, m.chunk_index) for m in self.metadata + metadata}
        if len(identities) != len(self.metadata) + len(metadata):
            raise ValueError("Duplicate document/chunk identity.")
        if any(self.document_exists(m.document_id) for m in metadata):
            raise ValueError("This document is already indexed.")
        index = faiss.clone_index(self.index)
        index.add(np.ascontiguousarray(vectors))
        self.save(index, self.metadata + metadata)

    def search(self, query_embedding, k=5):
        if not self.count():
            return []
        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        if query.shape[1] != self.index.d or not np.isfinite(query).all() or k < 1:
            raise ValueError("Invalid query vector or top-k value.")
        if not np.isclose(np.linalg.norm(query), 1, atol=1e-3):
            raise ValueError("Query embedding must be normalized.")
        scores, indices = self.index.search(query, min(k, self.count()))
        return [SearchResult(float(score), self.metadata[int(position)])
                for score, position in zip(scores[0], indices[0])
                if 0 <= position < len(self.metadata)]

    def count(self):
        return self.index.ntotal

    def save(self, index=None, metadata=None):
        index = self.index if index is None else index
        metadata = self.metadata if metadata is None else metadata
        if index.ntotal != len(metadata):
            raise ValueError("Vector/metadata count mismatch; nothing was saved.")
        raw = faiss.serialize_index(index).tobytes()
        manifest = {
            "version": 1,
            "embedding": EMBEDDING_CONFIG,
            "index_sha256": hashlib.sha256(raw).hexdigest(),
            "metadata": [asdict(chunk) for chunk in metadata],
        }
        # One archive binds both parts; only a complete archive replaces the prior snapshot.
        self.path.parent.mkdir(exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path.parent, suffix=".tmp", delete=False) as file:
                temporary = Path(file.name)
                with zipfile.ZipFile(file, "w") as archive:
                    archive.writestr("index.faiss", raw)
                    archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False))
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.index, self.metadata = index, metadata

    def load(self):
        try:
            with zipfile.ZipFile(self.path) as archive:
                raw = archive.read("index.faiss")
                manifest = json.loads(archive.read("manifest.json"))
            if manifest["version"] != 1 or manifest["embedding"] != EMBEDDING_CONFIG:
                raise ValueError("Unsupported snapshot or incompatible embedding configuration.")
            if hashlib.sha256(raw).hexdigest() != manifest["index_sha256"]:
                raise ValueError("Index checksum mismatch.")
            index = faiss.deserialize_index(np.frombuffer(raw, dtype="uint8"))
            metadata = [ChunkMetadata(**chunk) for chunk in manifest["metadata"]]
            if (not isinstance(index, faiss.IndexFlatIP) or index.d != EMBEDDING_CONFIG["dimension"]
                    or index.metric_type != faiss.METRIC_INNER_PRODUCT or index.ntotal != len(metadata)):
                raise ValueError("Index type, dimension, or metadata count mismatch.")
            identities = {(m.document_id, m.chunk_index) for m in metadata}
            if len(identities) != len(metadata) or any(
                not isinstance(m.document_id, str) or len(m.document_id) != 64
                or any(c not in "0123456789abcdef" for c in m.document_id)
                or not isinstance(m.chunk_index, int) or m.chunk_index < 0
                or not isinstance(m.page_number, int) or m.page_number < 1
                or not isinstance(m.text, str) or not m.text.strip()
                or not isinstance(m.document_name, str)
                for m in metadata
            ):
                raise ValueError("Invalid or duplicate chunk metadata.")
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, zipfile.BadZipFile) as error:
            raise ValueError(
                f"Cannot load knowledge base: {error} Preserve {self.path}; restore a trusted backup "
                "or move it aside and re-ingest PDFs. Nothing was reset."
            ) from error
        self.index, self.metadata = index, metadata

    def document_exists(self, document_id):
        return any(chunk.document_id == document_id for chunk in self.metadata)

    def get_documents(self):
        documents = {}
        for chunk in self.metadata:
            document = documents.setdefault(chunk.document_id, {
                "document_name": chunk.document_name, "chunk_count": 0,
            })
            document["chunk_count"] += 1
        return documents

    def delete_document(self, document_id):
        positions = [i for i, chunk in enumerate(self.metadata) if chunk.document_id != document_id]
        if len(positions) == self.count():
            return False
        index = faiss.IndexFlatIP(self.index.d)
        if positions:
            index.add(np.array([self.index.reconstruct(i) for i in positions], dtype="float32"))
        self.save(index, [self.metadata[i] for i in positions])
        return True
