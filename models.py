from dataclasses import dataclass


@dataclass
class ChunkMetadata:

    chunk_id: int

    document_id: str

    chunk_index: int

    document_name: str

    page_number: int

    text: str


@dataclass
class SearchResult:

    score: float

    metadata: ChunkMetadata