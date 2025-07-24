import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from src.api.rag_service import RAGService
from src.api.config import settings


@pytest.fixture
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_rag_service():
    with patch('src.api.rag_service.EmbedderFactory') as mock_embedder_factory, \
         patch('src.api.rag_service.ChromaVectorStore') as mock_vector_store, \
         patch('src.api.rag_service.OpenAI') as mock_openai, \
         patch('src.api.rag_service.DocumentLoader') as mock_doc_loader, \
         patch('src.api.rag_service.TextSplitter') as mock_text_splitter:
        
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder_factory.create_embedder.return_value = mock_embedder
        
        mock_vector = Mock()
        mock_vector_store.return_value = mock_vector
        
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_loader = Mock()
        mock_doc_loader.return_value = mock_loader
        
        mock_splitter = Mock()
        mock_text_splitter.return_value = mock_splitter
        
        service = RAGService()
        
        # Attach mocks to service for easy access in tests
        service._mock_embedder = mock_embedder
        service._mock_vector_store = mock_vector
        service._mock_openai_client = mock_client
        service._mock_doc_loader = mock_loader
        service._mock_text_splitter = mock_splitter
        
        yield service


def test_rag_service_initialization(mock_rag_service):
    """Test that RAGService initializes all components correctly"""
    service = mock_rag_service
    
    # Verify components are initialized
    assert hasattr(service, 'embedder')
    assert hasattr(service, 'vector_store')
    assert hasattr(service, 'document_loader')
    assert hasattr(service, 'text_splitter')


def test_ingest_documents_success(mock_rag_service, temp_dir):
    """Test successful document ingestion"""
    service = mock_rag_service
    
    # Create test documents
    doc1_path = Path(temp_dir) / "doc1.txt"
    doc1_path.write_text("Test document 1 content")
    
    doc2_path = Path(temp_dir) / "doc2.txt"
    doc2_path.write_text("Test document 2 content")
    
    # Mock document loader response
    from src.ingestion.document_loader import Document
    mock_documents = [
        Document(content="Test document 1 content", metadata={"file_name": "doc1.txt"}, source=str(doc1_path)),
        Document(content="Test document 2 content", metadata={"file_name": "doc2.txt"}, source=str(doc2_path))
    ]
    service._mock_doc_loader.load_documents.return_value = mock_documents
    
    # Mock text splitter response
    from src.ingestion.text_splitter import TextChunk
    mock_chunks = [
        TextChunk(content="Test document 1 content", metadata={"chunk_index": 0}, start_index=0, end_index=20),
        TextChunk(content="Test document 2 content", metadata={"chunk_index": 0}, start_index=0, end_index=20)
    ]
    service._mock_text_splitter.split_text.return_value = mock_chunks
    
    # Mock embedder for sentence transformer mode
    with patch.object(settings, 'embedder_type', 'sentence-transformer'):
        result = service.ingest_documents(temp_dir)
    
    # Verify results
    assert result['documents_processed'] == 2
    assert result['chunks_created'] == 4  # 2 chunks per document
    assert result['errors'] == []
    
    # Verify method calls
    service._mock_doc_loader.load_documents.assert_called_once_with(temp_dir)
    assert service._mock_text_splitter.split_text.call_count == 2
    service._mock_vector_store.add_documents.assert_called()


def test_ingest_documents_with_openai_embeddings(mock_rag_service, temp_dir):
    """Test document ingestion with OpenAI embeddings"""
    service = mock_rag_service
    
    # Create test document
    doc_path = Path(temp_dir) / "doc.txt"
    doc_path.write_text("Test document content")
    
    # Mock responses
    from src.ingestion.document_loader import Document
    mock_doc = Document(content="Test content", metadata={"file_name": "doc.txt"}, source=str(doc_path))
    service._mock_doc_loader.load_documents.return_value = [mock_doc]
    
    from src.ingestion.text_splitter import TextChunk
    mock_chunk = TextChunk(content="Test content", metadata={"chunk_index": 0}, start_index=0, end_index=12)
    service._mock_text_splitter.split_text.return_value = [mock_chunk]
    
    # Mock embeddings
    service._mock_embedder.embed_texts.return_value = [[0.1, 0.2, 0.3]]
    
    # Test with OpenAI embeddings
    with patch.object(settings, 'embedder_type', 'openai'):
        result = service.ingest_documents(temp_dir)
    
    # Verify OpenAI embedding path was used
    service._mock_embedder.embed_texts.assert_called_once()
    service._mock_vector_store.add_embeddings.assert_called_once()


def test_ingest_documents_with_errors(mock_rag_service, temp_dir):
    """Test error handling during document ingestion"""
    service = mock_rag_service
    
    # Mock an error during document loading
    service._mock_doc_loader.load_documents.side_effect = Exception("Loading failed")
    
    result = service.ingest_documents(temp_dir)
    
    # Verify error handling
    assert result['documents_processed'] == 0
    assert result['chunks_created'] == 0
    assert len(result['errors']) == 1
    assert "Loading failed" in result['errors'][0]


def test_search_with_sentence_transformer(mock_rag_service):
    """Test search functionality with sentence transformer"""
    service = mock_rag_service
    
    # Mock search results
    from src.storage.vector_store import SearchResult
    mock_results = [
        SearchResult(content="Test result", metadata={"source": "test.txt"}, score=0.9, id="1")
    ]
    service._mock_vector_store.search.return_value = mock_results
    
    # Test search
    with patch.object(settings, 'embedder_type', 'sentence-transformer'):
        results = service.search("test query", k=5)
    
    # Verify
    assert len(results) == 1
    assert results[0].content == "Test result"
    service._mock_vector_store.search.assert_called_once_with("test query", 5, None)


def test_search_with_openai_embeddings(mock_rag_service):
    """Test search functionality with OpenAI embeddings"""
    service = mock_rag_service
    
    # Mock embedding and search results
    service._mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    from src.storage.vector_store import SearchResult
    mock_results = [
        SearchResult(content="Test result", metadata={"source": "test.txt"}, score=0.9, id="1")
    ]
    service._mock_vector_store.search_by_embedding.return_value = mock_results
    
    # Test search
    with patch.object(settings, 'embedder_type', 'openai'):
        results = service.search("test query", k=3)
    
    # Verify
    service._mock_embedder.embed_query.assert_called_once_with("test query")
    service._mock_vector_store.search_by_embedding.assert_called_once_with([0.1, 0.2, 0.3], 3, None)


def test_chat_with_context(mock_rag_service):
    """Test chat functionality with context"""
    service = mock_rag_service
    
    # Mock search results
    from src.storage.vector_store import SearchResult
    mock_results = [
        SearchResult(content="Relevant content", metadata={"source": "test.txt"}, score=0.9, id="1")
    ]
    service._mock_vector_store.search.return_value = mock_results
    
    # Mock OpenAI response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "AI response based on context"
    service._mock_openai_client.chat.completions.create.return_value = mock_response
    
    # Set the openai_client attribute properly
    service.openai_client = service._mock_openai_client
    
    # Test chat with mocked settings
    with patch.object(service, 'search', return_value=mock_results):
        result = service.chat("What is this about?", k=3, use_context=True)
    
    # Verify
    assert result['message'] == "What is this about?"
    assert result['response'] == "AI response based on context"
    assert "test.txt" in result['sources']
    assert result['context_used'] is True


def test_chat_without_openai_key(mock_rag_service):
    """Test chat functionality without OpenAI API key"""
    service = mock_rag_service
    service.openai_client = None
    
    # Mock the search method to return empty results for this test
    service._mock_vector_store.search.return_value = []
    
    result = service.chat("Test message")
    
    # Verify fallback response
    assert "OpenAI API key not configured" in result['response']
    assert result['context_used'] is False


def test_get_document_count(mock_rag_service):
    """Test getting document count"""
    service = mock_rag_service
    service._mock_vector_store.count.return_value = 42
    
    count = service.get_document_count()
    
    assert count == 42
    service._mock_vector_store.count.assert_called_once()