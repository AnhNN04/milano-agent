import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...application.services.document_processing_service import (
    DocumentProcessingService,
)
from ...application.services.stock_analysis_service import StockAnalysisService
from ...shared.exceptions.domain_exceptions import (
    DocumentProcessingError,
    VectorStoreError,
)
from ...shared.logging.logger import Logger
from ...shared.session.redis_session_manager import (
    LangChainRedisSessionManager,
)
from ..dependencies.service import (
    get_document_processing_service,
    get_service_manager,
    get_session_manager,
    get_stock_analysis_service,
)
from ..models.requests import DocumentUploadRequest, QueryRequest
from ..models.responses import AgentResponse, DocumentProcessResponse

logger = Logger.get_logger(__name__)

# Application version
APP_VERSION = "1.0.0"
START_TIME = time.time()

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/query", response_model=AgentResponse)
async def query_agent(
    request: QueryRequest,
    stock_analysis_service: StockAnalysisService = Depends(
        get_stock_analysis_service
    ),
):
    """Process user query using StockAnalysisService with session and caching support."""
    start_time = time.time()
    try:
        logger.info(f"Processing query for session: {request.session_id}")

        result = await stock_analysis_service.analyze(
            query=request.query, session_id=request.session_id
        )

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Query processed in {elapsed_ms:.2f} ms")

        return AgentResponse(
            answer=result["answer"],
            success=result["metadata"]["success"],
            steps=result["metadata"]["steps"],
            tools_used=result["metadata"]["tools_used"],
            intermediate_results=result["metadata"]["intermediate_results"],
            session_id=result["metadata"]["session_id"],
            cached=result["metadata"].get("cached", False),
            query_hash=result["metadata"].get("query_hash"),
        )

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Query processing failed in {elapsed_ms:.2f} ms: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}",
        )


@router.post("/load-documents", response_model=DocumentProcessResponse)
async def load_documents(
    request: DocumentUploadRequest,
    doc_service: DocumentProcessingService = Depends(
        get_document_processing_service
    ),
):
    """Load and process documents from S3 into vector store."""
    start_time = time.time()
    try:
        logger.info(f"Processing {len(request.s3_keys)} documents")

        result = await doc_service.process_documents(s3_keys=request.s3_keys)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Loaded {result['processed_documents']} documents in {elapsed_ms:.2f} ms"
        )

        return DocumentProcessResponse(
            success=True,
            processed_documents=result["processed_documents"],
            total_chunks=result["total_chunks"],
            failed_documents=result["failed_documents"],
            processing_time=result["processing_time"],
        )

    except DocumentProcessingError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Document processing failed in {elapsed_ms:.2f} ms: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )
    except VectorStoreError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"Vector store error in {elapsed_ms:.2f} ms: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector store unavailable: {str(e)}",
        )
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Unexpected error in document processing after {elapsed_ms:.2f} ms: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during document processing",
        )


@router.get("/documents")
async def list_documents(
    prefix: Optional[str] = Query(
        None, description="Filter documents by prefix"
    ),
    limit: Optional[int] = Query(
        100, description="Maximum number of documents to return", ge=1, le=1000
    ),
    doc_service: DocumentProcessingService = Depends(
        get_document_processing_service
    ),
):
    """List available documents in S3 with optional filtering."""
    start_time = time.time()
    try:
        logger.info(f"Listing documents with prefix: {prefix}, limit: {limit}")

        documents = await doc_service.list_available_documents(prefix)

        # Apply limit
        if limit and len(documents) > limit:
            documents = documents[:limit]

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Listed {len(documents)} documents in {elapsed_ms:.2f} ms"
        )

        return {
            "documents": documents,
            "total_count": len(documents),
            "prefix": prefix,
            "limit": limit,
        }

    except DocumentProcessingError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Failed to list documents in {elapsed_ms:.2f} ms: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Failed to list documents in {elapsed_ms:.2f} ms: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document list",
        )


@router.post("/session/create")
async def create_session(
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    session_manager: LangChainRedisSessionManager = Depends(
        get_session_manager
    ),
):
    """Create a new user session with optional session_id and metadata."""
    start_time = time.time()
    try:
        if session_id:
            await session_manager.create_session_with_id(session_id, metadata)
        else:
            session_id = await session_manager.create_session(metadata)

        session_data = await session_manager.get_session(session_id)

        # Convert timestamps to datetime strings in ISO 8601 format
        created_at_dt = datetime.fromtimestamp(
            session_data.created_at, tz=timezone.utc
        )
        expires_at_dt = datetime.fromtimestamp(
            session_data.created_at + session_manager.default_ttl,
            tz=timezone.utc,
        )

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Created new session {session_id} in {elapsed_ms:.2f} ms")

        return {
            "session_id": session_id,
            "created_at": created_at_dt.isoformat(),
            "expires_at": expires_at_dt.isoformat(),
            "metadata": session_data.metadata,
            "message": "Session created successfully",
        }

    except ValueError as ve:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Failed to create session in {elapsed_ms:.2f} ms: {str(ve)}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve)
        )
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Failed to create session in {elapsed_ms:.2f} ms: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session",
        )


@router.get("/session/{session_id}")
async def get_session_info(
    session_id: str,
    session_manager: LangChainRedisSessionManager = Depends(
        get_session_manager
    ),
):
    """Get session information and metadata."""
    start_time = time.time()
    try:
        session = await session_manager.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or expired",
            )

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Retrieved session info for {session_id} in {elapsed_ms:.2f} ms"
        )

        return {
            "session_id": session.session_id,
            "created_at": datetime.fromtimestamp(
                session.created_at, tz=timezone.utc
            ),
            "last_accessed": datetime.fromtimestamp(
                session.created_at + session_manager.default_ttl,
                tz=timezone.utc,
            ),
            "metadata": session.metadata,
        }

    except HTTPException:
        raise
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Failed to get session info in {elapsed_ms:.2f} ms: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session information",
        )


@router.get("/session/{session_id}/history")
async def get_session_history(
    session_id: str,
    limit: Optional[int] = Query(
        50, description="Maximum number of messages to return", ge=1, le=1000
    ),
    offset: Optional[int] = Query(
        0, description="Number of messages to skip", ge=0
    ),
    stock_analysis_service: StockAnalysisService = Depends(
        get_stock_analysis_service
    ),
):
    """Get conversation history for a session with pagination."""
    start_time = time.time()
    try:
        history = await stock_analysis_service.get_session_history(session_id)

        if history is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or expired",
            )

        # Apply pagination
        total_messages = len(history)
        paginated_history = history[offset : offset + limit] if history else []

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Retrieved {len(paginated_history)} messages for session {session_id} in {elapsed_ms:.2f} ms"
        )

        return {
            "session_id": session_id,
            "history": paginated_history,
            "pagination": {
                "total": total_messages,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_messages,
                "next_offset": (
                    offset + limit if offset + limit < total_messages else None
                ),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Failed to get session history in {elapsed_ms:.2f} ms: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session history",
        )


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    session_manager: LangChainRedisSessionManager = Depends(
        get_session_manager
    ),
):
    """Delete a specific session and all its data."""
    start_time = time.time()
    try:
        session = await session_manager.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )

        await session_manager.delete_session(session_id)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Deleted session {session_id} in {elapsed_ms:.2f} ms")

        return {"message": f"Session {session_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Failed to delete session {session_id} in {elapsed_ms:.2f} ms: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session",
        )


@router.get("/health", response_model=Dict[str, Any])
async def health_check(service_manager=Depends(get_service_manager)):
    """Check the health of the API and underlying services."""
    start_time = time.time()
    logger.info("Received health check request")

    try:
        from ..dependencies.service import _session_manager

        stats = service_manager.get_stats()
        uptime = time.time() - START_TIME
        health_status = {
            "status": "healthy",
            "version": APP_VERSION,
            "uptime_seconds": round(uptime, 2),
            "services": stats,
            "session_manager_initialized": _session_manager is not None,
        }
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Health check response in {elapsed_ms:.2f} ms: {health_status}"
        )
        return health_status
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Health check failed in {elapsed_ms:.2f} ms: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}",
        )
