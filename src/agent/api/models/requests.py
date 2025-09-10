from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for agent queries"""

    query: str = Field(..., description="User query about Vietnamese stocks")
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation tracking"
    )
    max_steps: Optional[int] = Field(10, description="Maximum reasoning steps")


class DocumentUploadRequest(BaseModel):
    """Request model for document processing"""

    s3_keys: list[str] = Field(
        ..., description="List of S3 document keys to process"
    )
    force_reprocess: bool = Field(
        False, description="Force reprocessing of existing documents"
    )
