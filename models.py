from dataclasses import dataclass


@dataclass
class ChunkMetadata:
    chunk_id: int
    chunk_index: int
    document_name: str
    text: str