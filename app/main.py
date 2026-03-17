"""
FastAPI application for the RAG (Retrieval-Augmented Generation) pipeline.

Provides endpoints for document ingestion, querying, and health monitoring.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from app.rag_core import RAGPipeline

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
log_file = os.getenv("LOG_FILE", "./logs/rag_pipeline.log")

# Create logs directory if it doesn't exist
os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)

logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=os.getenv("API_TITLE", "RAG Pipeline API"),
    version=os.getenv("API_VERSION", "1.0.0"),
    description="Production-grade RAG pipeline with document ingestion and intelligent querying",
)

# Configure CORS
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Pipeline
try:
    rag_pipeline = RAGPipeline(
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        chroma_db_path=os.getenv("CHROMA_DB_PATH", "./vectorstore/chroma_db"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "mistral.codestral-embed"),
        llm_model=os.getenv("LLM_MODEL", "mistral-large-latest"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
    )
    logger.info("RAG Pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG Pipeline: {str(e)}")
    raise

# Create data directory for uploads
UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for querying the RAG pipeline."""

    question: str
    n_results: int = 5


class QueryResponse(BaseModel):
    """Response model for query results."""

    success: bool
    question: str
    answer: Optional[str] = None
    sources: Optional[list] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str
    database_status: str
    document_count: int


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for monitoring service status.

    Returns:
        HealthResponse with service status and statistics.
    """
    try:
        stats = rag_pipeline.vector_store.get_collection_stats()
        logger.info("Health check performed successfully")
        return HealthResponse(
            status="healthy",
            version=os.getenv("API_VERSION", "1.0.0"),
            database_status="connected",
            document_count=stats["document_count"],
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()) -> dict:
    """
    Document ingestion endpoint.

    Accepts PDF and DOCX files, chunks them, generates embeddings, and stores in ChromaDB.

    Args:
        file: Uploaded document file (PDF or DOCX).
        background_tasks: Background task manager for cleanup.

    Returns:
        Dictionary with ingestion status and results.

    Raises:
        HTTPException: If file format is unsupported or ingestion fails.
    """
    try:
        # Validate file format
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in [".pdf", ".docx"]:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Only PDF and DOCX files are supported.",
            )

        # Validate file size
        max_size = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50")) * 1024 * 1024
        file_content = await file.read()
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum allowed size of {os.getenv('MAX_UPLOAD_SIZE_MB', '50')}MB",
            )

        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(file_content)

        logger.info(f"Received file for ingestion: {file.filename}")

        # Ingest document
        document_name = Path(file.filename).stem
        result = rag_pipeline.ingest_document(str(file_path), document_name)

        # Schedule file cleanup
        background_tasks.add_task(cleanup_file, str(file_path))

        logger.info(f"Document ingestion completed: {document_name}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Query endpoint for RAG-based question answering.

    Takes a user question, retrieves relevant document chunks, and generates an answer.

    Args:
        request: QueryRequest containing the question and number of results.

    Returns:
        QueryResponse with the generated answer and source chunks.

    Raises:
        HTTPException: If query processing fails.
    """
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        logger.info(f"Processing query: {request.question}")
        result = rag_pipeline.query(request.question, request.n_results)

        if not result.get("success"):
            logger.error(f"Query failed: {result.get('error')}")
            raise HTTPException(status_code=500, detail=result.get("error", "Query processing failed"))

        logger.info("Query processed successfully")
        return QueryResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/stats")
async def get_stats() -> dict:
    """
    Get statistics about the RAG pipeline.

    Returns:
        Dictionary with pipeline statistics.
    """
    try:
        stats = rag_pipeline.vector_store.get_collection_stats()
        logger.info("Statistics retrieved successfully")
        return {
            "success": True,
            "stats": stats,
        }
    except Exception as e:
        logger.error(f"Failed to retrieve statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")


@app.get("/")
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": os.getenv("API_TITLE", "RAG Pipeline API"),
        "version": os.getenv("API_VERSION", "1.0.0"),
        "description": "Production-grade RAG pipeline with document ingestion and intelligent querying",
        "endpoints": {
            "health": "/health",
            "ingest": "/ingest",
            "query": "/query",
            "stats": "/stats",
            "docs": "/docs",
        },
    }


# Helper functions
def cleanup_file(file_path: str) -> None:
    """
    Clean up uploaded file after processing.

    Args:
        file_path: Path to the file to delete.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {str(e)}")


# Application startup and shutdown
@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    logger.info("RAG Pipeline API starting up")
    logger.info(f"API Title: {os.getenv('API_TITLE', 'RAG Pipeline API')}")
    logger.info(f"API Version: {os.getenv('API_VERSION', '1.0.0')}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    logger.info("RAG Pipeline API shutting down")


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    logger.info(f"Starting RAG Pipeline API on {host}:{port}")
    uvicorn.run(app, host=host, port=port, debug=debug)
