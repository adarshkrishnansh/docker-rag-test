import pytest
import tempfile
import shutil
from src.storage.vector_store import ChromaVectorStore, SearchResult


@pytest.fixture
def temp_db_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def vector_store(temp_db_dir):
    return ChromaVectorStore(
        collection_name="test_collection",
        persist_directory=temp_db_dir
    )


def test_add_and_search_documents(vector_store):
    # Add documents
    documents = [
        "Machine learning is a subset of AI",
        "Python is a popular programming language",
        "Data science uses machine learning"
    ]
    metadatas = [
        {"topic": "ML", "id": 1},
        {"topic": "Programming", "id": 2},
        {"topic": "Data Science", "id": 3}
    ]
    
    ids = vector_store.add_documents(documents, metadatas)
    
    assert len(ids) == 3
    
    # Search for documents
    results = vector_store.search("machine learning", k=2)
    
    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)
    
    # Check that ML-related documents are returned
    topics = [r.metadata["topic"] for r in results]
    assert "ML" in topics or "Data Science" in topics


def test_add_embeddings_directly(vector_store):
    # Create dummy embeddings
    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ]
    documents = ["Doc 1", "Doc 2", "Doc 3"]
    metadatas = [{"id": i} for i in range(3)]
    
    ids = vector_store.add_embeddings(embeddings, documents, metadatas)
    
    assert len(ids) == 3
    assert vector_store.count() == 3


def test_search_with_filter(vector_store):
    # Add documents with different categories
    documents = [
        "Python programming tutorial",
        "Java programming guide",
        "Python data analysis"
    ]
    metadatas = [
        {"language": "Python", "type": "tutorial"},
        {"language": "Java", "type": "guide"},
        {"language": "Python", "type": "analysis"}
    ]
    
    vector_store.add_documents(documents, metadatas)
    
    # Search with filter
    results = vector_store.search(
        "programming",
        k=3,
        filter={"language": "Python"}
    )
    
    # Should only return Python documents
    assert all(r.metadata["language"] == "Python" for r in results)


def test_delete_documents(vector_store):
    # Add documents
    documents = ["Doc 1", "Doc 2", "Doc 3"]
    metadatas = [{"id": i} for i in range(3)]
    ids = vector_store.add_documents(documents, metadatas)
    
    initial_count = vector_store.count()
    assert initial_count == 3
    
    # Delete one document
    vector_store.delete([ids[0]])
    
    assert vector_store.count() == 2
    
    # Verify deleted document is not in search results
    results = vector_store.search("Doc", k=3)
    assert len(results) == 2
    assert not any(r.id == ids[0] for r in results)


def test_update_documents(vector_store):
    # Add a document
    documents = ["Original content"]
    metadatas = [{"version": 1}]
    ids = vector_store.add_documents(documents, metadatas)
    
    # Update the document
    vector_store.update(
        ids=ids,
        documents=["Updated content"],
        metadatas=[{"version": 2}]
    )
    
    # Verify update
    doc_data = vector_store.get(ids)
    assert doc_data['documents'][0] == "Updated content"
    assert doc_data['metadatas'][0]['version'] == 2


def test_reset_collection(vector_store):
    # Add documents
    documents = ["Doc 1", "Doc 2", "Doc 3"]
    metadatas = [{"id": i} for i in range(3)]
    vector_store.add_documents(documents, metadatas)
    
    assert vector_store.count() == 3
    
    # Reset collection
    vector_store.reset()
    
    assert vector_store.count() == 0