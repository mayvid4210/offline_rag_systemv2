import faiss


class VectorStore:

    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings):
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings of dimension {self.dimension}, got {embeddings.shape[1]}"
            )

        self.index.add(embeddings)

    def count(self):
        return self.index.ntotal