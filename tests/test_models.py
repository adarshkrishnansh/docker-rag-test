import pytest
from pydantic import ValidationError
from datetime import datetime
from src.api.models import (
    DocumentUpload, DocumentResponse, QueryRequest, QueryResult, QueryResponse,
    ChatRequest, ChatResponse, HealthResponse, IngestRequest, IngestResponse
)


class TestDocumentUpload:
    def test_valid_document_upload(self):
        """Test valid document upload creation"""
        doc = DocumentUpload(
            content="Test document content",
            metadata={"author": "Test Author", "type": "text"},
            source="test.txt"
        )
        
        assert doc.content == "Test document content"
        assert doc.metadata == {"author": "Test Author", "type": "text"}
        assert doc.source == "test.txt"
    
    def test_document_upload_default_metadata(self):
        """Test document upload with default empty metadata"""
        doc = DocumentUpload(
            content="Test content",
            source="test.txt"
        )
        
        assert doc.metadata == {}
    
    def test_document_upload_validation_empty_content(self):
        """Test that empty content is still valid (might be intentional)"""
        doc = DocumentUpload(
            content="",
            source="test.txt"
        )
        
        assert doc.content == ""


class TestDocumentResponse:
    def test_valid_document_response(self):
        """Test valid document response creation"""
        response = DocumentResponse(
            id="doc-123",
            source="test.txt",
            metadata={"size": 1024}
        )
        
        assert response.id == "doc-123"
        assert response.source == "test.txt"
        assert response.metadata == {"size": 1024}
        assert isinstance(response.created_at, datetime)
    
    def test_document_response_with_custom_datetime(self):
        """Test document response with custom datetime"""
        custom_time = datetime(2023, 1, 1, 12, 0, 0)
        response = DocumentResponse(
            id="doc-123",
            source="test.txt",
            metadata={},
            created_at=custom_time
        )
        
        assert response.created_at == custom_time


class TestQueryRequest:
    def test_valid_query_request(self):
        """Test valid query request creation"""
        request = QueryRequest(
            query="What is machine learning?",
            k=5,
            filter={"type": "pdf"}
        )
        
        assert request.query == "What is machine learning?"
        assert request.k == 5
        assert request.filter == {"type": "pdf"}
    
    def test_query_request_defaults(self):
        """Test query request with default values"""
        request = QueryRequest(query="test query")
        
        assert request.query == "test query"
        assert request.k == 5
        assert request.filter is None
    
    def test_query_request_k_validation(self):
        """Test k parameter validation"""
        # Valid k values
        for k in [1, 5, 10, 20]:
            request = QueryRequest(query="test", k=k)
            assert request.k == k
        
        # Invalid k values
        with pytest.raises(ValidationError):
            QueryRequest(query="test", k=0)
        
        with pytest.raises(ValidationError):
            QueryRequest(query="test", k=-1)
        
        with pytest.raises(ValidationError):
            QueryRequest(query="test", k=21)


class TestQueryResult:
    def test_valid_query_result(self):
        """Test valid query result creation"""
        result = QueryResult(
            content="Test content",
            metadata={"source": "test.txt", "chunk": 1},
            score=0.95,
            source="test.txt"
        )
        
        assert result.content == "Test content"
        assert result.metadata == {"source": "test.txt", "chunk": 1}
        assert result.score == 0.95
        assert result.source == "test.txt"


class TestQueryResponse:
    def test_valid_query_response(self):
        """Test valid query response creation"""
        results = [
            QueryResult(
                content="Result 1",
                metadata={"source": "doc1.txt"},
                score=0.9,
                source="doc1.txt"
            ),
            QueryResult(
                content="Result 2",
                metadata={"source": "doc2.txt"},
                score=0.8,
                source="doc2.txt"
            )
        ]
        
        response = QueryResponse(
            query="test query",
            results=results,
            total_results=2
        )
        
        assert response.query == "test query"
        assert len(response.results) == 2
        assert response.total_results == 2
    
    def test_query_response_empty_results(self):
        """Test query response with no results"""
        response = QueryResponse(
            query="no results query",
            results=[],
            total_results=0
        )
        
        assert len(response.results) == 0
        assert response.total_results == 0


class TestChatRequest:
    def test_valid_chat_request(self):
        """Test valid chat request creation"""
        request = ChatRequest(
            message="What is AI?",
            k=3,
            use_context=True,
            temperature=0.7
        )
        
        assert request.message == "What is AI?"
        assert request.k == 3
        assert request.use_context is True
        assert request.temperature == 0.7
    
    def test_chat_request_defaults(self):
        """Test chat request with default values"""
        request = ChatRequest(message="test message")
        
        assert request.message == "test message"
        assert request.k == 5
        assert request.use_context is True
        assert request.temperature == 0.7
    
    def test_chat_request_k_validation(self):
        """Test k parameter validation in chat request"""
        # Valid k values
        for k in [1, 10, 20]:
            request = ChatRequest(message="test", k=k)
            assert request.k == k
        
        # Invalid k values
        with pytest.raises(ValidationError):
            ChatRequest(message="test", k=0)
        
        with pytest.raises(ValidationError):
            ChatRequest(message="test", k=21)
    
    def test_chat_request_temperature_validation(self):
        """Test temperature parameter validation"""
        # Valid temperature values
        for temp in [0.0, 0.5, 1.0, 1.5, 2.0]:
            request = ChatRequest(message="test", temperature=temp)
            assert request.temperature == temp
        
        # Invalid temperature values
        with pytest.raises(ValidationError):
            ChatRequest(message="test", temperature=-0.1)
        
        with pytest.raises(ValidationError):
            ChatRequest(message="test", temperature=2.1)


class TestChatResponse:
    def test_valid_chat_response(self):
        """Test valid chat response creation"""
        response = ChatResponse(
            message="What is AI?",
            response="AI is artificial intelligence...",
            sources=["doc1.txt", "doc2.txt"],
            context_used=True
        )
        
        assert response.message == "What is AI?"
        assert "artificial intelligence" in response.response
        assert response.sources == ["doc1.txt", "doc2.txt"]
        assert response.context_used is True
    
    def test_chat_response_no_sources(self):
        """Test chat response with no sources"""
        response = ChatResponse(
            message="General question",
            response="General answer",
            sources=[],
            context_used=False
        )
        
        assert len(response.sources) == 0
        assert response.context_used is False


class TestHealthResponse:
    def test_valid_health_response(self):
        """Test valid health response creation"""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            document_count=42
        )
        
        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.document_count == 42


class TestIngestRequest:
    def test_valid_ingest_request(self):
        """Test valid ingest request creation"""
        request = IngestRequest(directory_path="/app/data/documents")
        
        assert request.directory_path == "/app/data/documents"


class TestIngestResponse:
    def test_valid_ingest_response(self):
        """Test valid ingest response creation"""
        response = IngestResponse(
            documents_processed=5,
            chunks_created=25,
            errors=["Error processing file1.pdf"]
        )
        
        assert response.documents_processed == 5
        assert response.chunks_created == 25
        assert response.errors == ["Error processing file1.pdf"]
    
    def test_ingest_response_no_errors(self):
        """Test ingest response with no errors"""
        response = IngestResponse(
            documents_processed=3,
            chunks_created=15,
            errors=[]
        )
        
        assert len(response.errors) == 0


class TestModelSerialization:
    def test_model_to_dict(self):
        """Test model serialization to dictionary"""
        request = QueryRequest(
            query="test query",
            k=3,
            filter={"type": "pdf"}
        )
        
        data = request.model_dump()
        
        assert data["query"] == "test query"
        assert data["k"] == 3
        assert data["filter"] == {"type": "pdf"}
    
    def test_model_from_dict(self):
        """Test model creation from dictionary"""
        data = {
            "query": "test query",
            "k": 7,
            "filter": {"category": "science"}
        }
        
        request = QueryRequest(**data)
        
        assert request.query == "test query"
        assert request.k == 7
        assert request.filter == {"category": "science"}
    
    def test_model_json_serialization(self):
        """Test JSON serialization"""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            document_count=10
        )
        
        json_str = response.model_dump_json()
        assert '"status":"healthy"' in json_str
        assert '"document_count":10' in json_str
    
    def test_nested_model_serialization(self):
        """Test serialization of models with nested objects"""
        results = [
            QueryResult(
                content="Test",
                metadata={"source": "test.txt"},
                score=0.9,
                source="test.txt"
            )
        ]
        
        response = QueryResponse(
            query="test",
            results=results,
            total_results=1
        )
        
        data = response.model_dump()
        assert len(data["results"]) == 1
        assert data["results"][0]["content"] == "Test"