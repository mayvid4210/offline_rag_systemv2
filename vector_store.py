import json
from pathlib import Path
from dataclasses import asdict

import faiss

from models import ChunkMetadata


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

    def add(self, embeddings, metadata: list[ChunkMetadata]):

        if self.index is None:
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings of dimension {self.dimension}, got {embeddings.shape[1]}"
            )

        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def count(self):

        if self.index is None:
            return 0

        return self.index.ntotal

    def save(self):

        if self.index is None:
            return

        faiss.write_index(
            self.index,
            str(INDEX_PATH)
        )

        metadata = [
            asdict(chunk)
            for chunk in self.metadata
        ]

        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=4)

    def load(self):

        self.index = faiss.read_index(
            str(INDEX_PATH)
        )

        self.dimension = self.index.d

        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)

        self.metadata = [
            ChunkMetadata(**chunk)
            for chunk in metadata
        ]