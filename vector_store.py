import json
from pathlib import Path
from dataclasses import asdict

import faiss
import numpy as np

from models import ChunkMetadata, SearchResult


INDEX_DIR = Path("knowledge_base")
INDEX_DIR.mkdir(exist_ok=True)

INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"


class VectorStore:

    def __init__(self):

        self.index = None
        self.dimension = None
        self.metadata = []

        if INDEX_PATH.exists():
            self.load()


    # ------------------------------------
    # Add vectors and metadata
    # ------------------------------------

    def add(self, embeddings, metadata):

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        if self.index is None:

            self.dimension = embeddings.shape[1]

            self.index = faiss.IndexFlatIP(
                self.dimension
            )

        if embeddings.shape[1] != self.dimension:

            raise ValueError(
                f"Expected embeddings of dimension "
                f"{self.dimension}, got "
                f"{embeddings.shape[1]}"
            )

        self.index.add(embeddings)

        self.metadata.extend(metadata)


    # ------------------------------------
    # Search
    # ------------------------------------

    def search(self, query_embedding, k=5):

        if self.index is None:
            raise ValueError(
                "Vector store is empty."
            )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        ).reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            results.append(
                SearchResult(
                    score=float(score),
                    metadata=self.metadata[index]
                )
            )

        return results


    # ------------------------------------
    # Count chunks
    # ------------------------------------

    def count(self):

        if self.index is None:
            return 0

        return self.index.ntotal


    # ------------------------------------
    # Save
    # ------------------------------------

    def save(self):

        if self.index is None:

            if INDEX_PATH.exists():
                INDEX_PATH.unlink()

            if METADATA_PATH.exists():
                METADATA_PATH.unlink()

            return

        faiss.write_index(
            self.index,
            str(INDEX_PATH)
        )

        metadata = [
            asdict(chunk)
            for chunk in self.metadata
        ]

        with open(
            METADATA_PATH,
            "w"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4
            )


    # ------------------------------------
    # Load
    # ------------------------------------

    def load(self):

        self.index = faiss.read_index(
            str(INDEX_PATH)
        )

        self.dimension = self.index.d

        with open(
            METADATA_PATH,
            "r"
        ) as f:

            metadata = json.load(f)

        self.metadata = [
            ChunkMetadata(**chunk)
            for chunk in metadata
        ]


    # ------------------------------------
    # Check duplicate document
    # ------------------------------------

    def document_exists(self, document_id):

        return any(
            chunk.document_id == document_id
            for chunk in self.metadata
        )


    # ------------------------------------
    # Get all documents
    # ------------------------------------

    def get_documents(self):

        documents = {}

        for chunk in self.metadata:

            if chunk.document_id not in documents:

                documents[chunk.document_id] = {
                    "document_name": chunk.document_name,
                    "chunk_count": 0
                }

            documents[
                chunk.document_id
            ]["chunk_count"] += 1

        return documents


    # ------------------------------------
    # Delete document
    # ------------------------------------

    def delete_document(self, document_id):

        if self.index is None:
            return False

        indices_to_keep = []
        new_metadata = []

        for index, chunk in enumerate(self.metadata):

            if chunk.document_id != document_id:

                indices_to_keep.append(index)
                new_metadata.append(chunk)


        # Document not found
        if len(new_metadata) == len(self.metadata):
            return False


        # If no vectors remain
        if not indices_to_keep:

            self.index = None
            self.dimension = None
            self.metadata = []

            self.save()

            return True


        # Get vectors of documents we want to keep
        remaining_vectors = np.array(
            [
                self.index.reconstruct(index)
                for index in indices_to_keep
            ],
            dtype="float32"
        )


        # Create a new FAISS index
        self.index = faiss.IndexFlatIP(
            self.dimension
        )


        # Add remaining vectors
        self.index.add(
            remaining_vectors
        )


        # Update metadata
        self.metadata = new_metadata


        # Save updated knowledge base
        self.save()

        return True