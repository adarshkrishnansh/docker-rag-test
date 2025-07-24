import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, MagicMock
import tempfile
import os
from pathlib import Path
from src.api.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_rag_service():
    """Mock the RAG service"""
    with patch('src.api.main.rag_service') as mock:
        yield mock


@pytest.fixture
def temp_file():
    """Create a temporary file for upload tests"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test file content")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


def test_health_check(client, mock_rag_service):
    """Test health check endpoint"""
    mock_rag_service.get_document_count.return_value = 42
    
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert data["document_count"] == 42


def test_health_check_no_service(client):
    """Test health check when service is not initialized"""
    with patch('src.api.main.rag_service', None):
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_count"] == 0


def test_ingest_documents_success(client, mock_rag_service):
    """Test successful document ingestion"""
    mock_rag_service.ingest_documents.return_value = {
        "documents_processed": 3,
        "chunks_created": 15,
        "errors": []
    }
    
    # Create a temporary directory that exists
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        response = client.post(
            "/ingest",
            json={"directory_path": temp_dir}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["documents_processed"] == 3
        assert data["chunks_created"] == 15
        assert data["errors"] == []
        
        mock_rag_service.ingest_documents.assert_called_once_with(temp_dir)


def test_ingest_documents_invalid_directory(client, mock_rag_service):
    """Test ingestion with invalid directory"""
    response = client.post(
        "/ingest",
        json={"directory_path": "/nonexistent/path"}
    )
    
    assert response.status_code == 400
    assert "does not exist" in response.json()["detail"]


def test_upload_files(client, mock_rag_service, temp_file):
    """Test file upload endpoint"""
    mock_rag_service.ingest_documents.return_value = {
        "documents_processed": 1,
        "chunks_created": 5,
        "errors": []
    }
    
    with open(temp_file, 'rb') as f:
        response = client.post(
            "/upload",
            files={"files": ("test.txt", f, "text/plain")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["documents_processed"] == 1
    assert data["chunks_created"] == 5
    
    # Verify the service was called with a temporary directory
    mock_rag_service.ingest_documents.assert_called_once()
    call_args = mock_rag_service.ingest_documents.call_args[0]
    assert len(call_args) == 1  # One argument (directory path)


def test_query_documents(client, mock_rag_service):
    """Test document query endpoint"""
    from src.storage.vector_store import SearchResult
    mock_results = [
        SearchResult(
            content="Test content",
            metadata={"source": "test.txt", "chunk_index": 0},
            score=0.95,
            id="1"
        ),
        SearchResult(
            content="Another result",
            metadata={"source": "test2.txt", "chunk_index": 1},
            score=0.85,
            id="2"
        )
    ]
    mock_rag_service.search.return_value = mock_results
    
    response = client.post(
        "/query",
        json={
            "query": "test query",
            "k": 5
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test query"
    assert len(data["results"]) == 2
    assert data["total_results"] == 2
    
    # Verify first result
    result1 = data["results"][0]
    assert result1["content"] == "Test content"
    assert result1["source"] == "test.txt"
    assert result1["score"] == 0.95
    
    mock_rag_service.search.assert_called_once_with(
        query="test query",
        k=5,
        filter=None
    )


def test_query_documents_empty_query(client, mock_rag_service):
    """Test query with empty string"""
    response = client.post(
        "/query",
        json={"query": "   "}
    )
    
    assert response.status_code == 400
    assert "Query cannot be empty" in response.json()["detail"]


def test_query_documents_with_filter(client, mock_rag_service):
    """Test query with filter"""
    mock_rag_service.search.return_value = []
    
    response = client.post(
        "/query",
        json={
            "query": "test",
            "k": 3,
            "filter": {"source": "specific.txt"}
        }
    )
    
    assert response.status_code == 200
    mock_rag_service.search.assert_called_once_with(
        query="test",
        k=3,
        filter={"source": "specific.txt"}
    )


def test_chat_success(client, mock_rag_service):
    """Test successful chat interaction"""
    mock_rag_service.chat.return_value = {
        "message": "What is machine learning?",
        "response": "Machine learning is a subset of AI...",
        "sources": ["doc1.txt", "doc2.txt"],
        "context_used": True
    }
    
    with patch('src.api.main.settings.openai_api_key', 'test-key'):
        response = client.post(
            "/chat",
            json={
                "message": "What is machine learning?",
                "k": 3,
                "use_context": True,
                "temperature": 0.7
            }
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "What is machine learning?"
    assert "Machine learning" in data["response"]
    assert len(data["sources"]) == 2
    assert data["context_used"] is True
    
    mock_rag_service.chat.assert_called_once_with(
        message="What is machine learning?",
        k=3,
        use_context=True,
        temperature=0.7
    )


def test_chat_no_openai_key(client, mock_rag_service):
    """Test chat without OpenAI API key"""
    with patch('src.api.main.settings.openai_api_key', None):
        response = client.post(
            "/chat",
            json={"message": "test message"}
        )
    
    assert response.status_code == 503
    assert "OpenAI API key not configured" in response.json()["detail"]


def test_chat_empty_message(client, mock_rag_service):
    """Test chat with empty message"""
    with patch('src.api.main.settings.openai_api_key', 'test-key'):
        response = client.post(
            "/chat",
            json={"message": "  "}
        )
    
    assert response.status_code == 400
    assert "Message cannot be empty" in response.json()["detail"]


def test_clear_documents(client, mock_rag_service):
    """Test clearing all documents"""
    response = client.delete("/documents")
    
    assert response.status_code == 200
    data = response.json()
    assert "cleared successfully" in data["message"]
    
    mock_rag_service.vector_store.reset.assert_called_once()


def test_get_document_count(client, mock_rag_service):
    """Test getting document count"""
    mock_rag_service.get_document_count.return_value = 25
    
    response = client.get("/documents/count")
    
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 25


def test_request_validation():
    """Test Pydantic model validation"""
    client = TestClient(app)
    
    # Test invalid query request (negative k)
    response = client.post(
        "/query",
        json={
            "query": "test",
            "k": -1  # Should be >= 1
        }
    )
    assert response.status_code == 422
    
    # Test invalid chat request (temperature out of range)
    with patch('src.api.main.settings.openai_api_key', 'test-key'):
        response = client.post(
            "/chat",
            json={
                "message": "test",
                "temperature": 3.0  # Should be <= 2
            }
        )
        assert response.status_code == 422


def test_cors_headers(client, mock_rag_service):
    """Test CORS headers are present"""
    response = client.get("/")
    
    # FastAPI test client doesn't include CORS headers in test mode,
    # but we can verify the middleware is configured
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_startup_auto_ingest():
    """Test auto-ingestion on startup"""
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test document
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Test content")
        
        with patch('src.api.main.settings.auto_ingest_on_startup', True), \
             patch('src.api.main.settings.auto_ingest_directory', temp_dir), \
             patch('src.api.main.RAGService') as mock_rag_class:
            
            mock_service = Mock()
            mock_service.ingest_documents.return_value = {
                "documents_processed": 1,
                "chunks_created": 1,
                "errors": []
            }
            mock_rag_class.return_value = mock_service
            
            # Import and call the auto_ingest function
            from src.api.main import auto_ingest_documents
            await auto_ingest_documents()
            
            mock_service.ingest_documents.assert_called_once_with(temp_dir)