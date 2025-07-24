from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class DocumentUpload(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str


class DocumentResponse(BaseModel):
    id: str
    source: str
    metadata: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now)


class QueryRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=20)
    filter: Optional[Dict[str, Any]] = None


class QueryResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str


class QueryResponse(BaseModel):
    query: str
    results: List[QueryResult]
    total_results: int


class ChatRequest(BaseModel):
    message: str
    k: int = Field(default=5, ge=1, le=20)
    use_context: bool = True
    temperature: float = Field(default=0.7, ge=0, le=2)


class ChatResponse(BaseModel):
    message: str
    response: str
    sources: List[str]
    context_used: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    document_count: int


class IngestRequest(BaseModel):
    directory_path: str


class IngestResponse(BaseModel):
    documents_processed: int
    chunks_created: int
    errors: List[str]