from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    """Metadata for documents"""

    source: str  # S3 path or URL
    title: Optional[str] = None
    document_type: str  # pdf, docx, txt
    tags: List[str] = []
    language: str = "vi"  # Vietnamese by default
    chunk_index: str
    start_char: int
    end_char: int


class DocumentChunk(BaseModel):
    """Document chunk for vector storage"""

    content: str
    metadata: DocumentMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata.model_dump(),
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }
