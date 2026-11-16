from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_CONFIG = {
    "model": MODEL_NAME,
    "dimension": 384,
    "normalized": True,
    "query_instruction": "",
    "chunk_size": 1000,
    "chunk_overlap": 200,
}


class Embedder:
    def __init__(self):
        try:
            self.model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        except OSError as error:
            raise ValueError(
                "BGE is unavailable locally. Cache BAAI/bge-small-en-v1.5 during setup before running offline."
            ) from error

    def embed_documents(self, chunks):
        return self.model.encode(chunks, normalize_embeddings=True)

    def embed_query(self, query):
        return self.model.encode(query, normalize_embeddings=True)
