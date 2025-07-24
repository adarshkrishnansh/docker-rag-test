from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
from typing import List
import tempfile
import shutil

from src.api.models import (
    QueryRequest, QueryResponse, QueryResult,
    ChatRequest, ChatResponse,
    HealthResponse,
    IngestRequest, IngestResponse,
    DocumentUpload, DocumentResponse
)
from src.api.rag_service import RAGService
from src.api.config import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Global RAG service instance
rag_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global rag_service
    logger.info("Starting RAG service...")
    rag_service = RAGService()
    logger.info("RAG service started successfully")
    
    # Auto-ingest documents on startup
    await auto_ingest_documents()
    
    yield
    # Shutdown
    logger.info("Shutting down RAG service...")


async def auto_ingest_documents():
    """Automatically ingest documents from the configured directory on startup"""
    if not settings.auto_ingest_on_startup:
        logger.info("Auto-ingestion is disabled")
        return
        
    documents_path = settings.auto_ingest_directory
    
    if os.path.exists(documents_path) and os.listdir(documents_path):
        logger.info(f"Auto-ingesting documents from {documents_path}...")
        try:
            result = rag_service.ingest_documents(documents_path)
            logger.info(f"Auto-ingestion completed: {result['documents_processed']} documents, "
                       f"{result['chunks_created']} chunks created")
            if result['errors']:
                logger.warning(f"Auto-ingestion errors: {result['errors']}")
        except Exception as e:
            logger.error(f"Auto-ingestion failed: {e}")
    else:
        logger.info(f"No documents found in {documents_path} for auto-ingestion")


# Create FastAPI app
app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API for document search and Q&A",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        document_count=rag_service.get_document_count() if rag_service else 0
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    if not os.path.exists(request.directory_path):
        raise HTTPException(status_code=400, detail=f"Directory {request.directory_path} does not exist")
    
    result = rag_service.ingest_documents(request.directory_path)
    
    return IngestResponse(**result)


@app.post("/upload", response_model=IngestResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    # Create temporary directory for uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded files
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
        
        # Ingest the files
        result = rag_service.ingest_documents(temp_dir)
        
        return IngestResponse(**result)


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    search_results = rag_service.search(
        query=request.query,
        k=request.k,
        filter=request.filter
    )
    
    results = [
        QueryResult(
            content=result.content,
            metadata=result.metadata,
            score=result.score,
            source=result.metadata.get('source', 'Unknown')
        )
        for result in search_results
    ]
    
    return QueryResponse(
        query=request.query,
        results=results,
        total_results=len(results)
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
        )
    
    result = rag_service.chat(
        message=request.message,
        k=request.k,
        use_context=request.use_context,
        temperature=request.temperature
    )
    
    return ChatResponse(**result)


@app.delete("/documents")
async def clear_documents():
    rag_service.vector_store.reset()
    return {"message": "All documents cleared successfully"}


@app.get("/documents/count")
async def get_document_count():
    count = rag_service.get_document_count()
    return {"count": count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development"
    )