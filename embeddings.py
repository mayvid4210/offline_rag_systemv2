from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(self):
        self.model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )

    def embed_documents(self, chunks):
        """
        Generate embeddings for document chunks.
        """

        embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True
        )

        return embeddings

    def embed_query(self, query):
        """
        Generate an embedding for a user query.
        """

        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        return embedding