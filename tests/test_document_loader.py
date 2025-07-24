import pytest
import tempfile
from pathlib import Path
from src.ingestion.document_loader import DocumentLoader, Document


@pytest.fixture
def document_loader():
    return DocumentLoader()


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


def test_load_text_file(document_loader, temp_dir):
    # Create a test text file
    test_file = Path(temp_dir) / "test.txt"
    test_content = "This is a test document."
    test_file.write_text(test_content)
    
    # Load the document
    doc = document_loader.load_document(str(test_file))
    
    assert isinstance(doc, Document)
    assert doc.content == test_content
    assert doc.metadata['file_name'] == "test.txt"
    assert doc.metadata['file_type'] == ".txt"


def test_load_markdown_file(document_loader, temp_dir):
    # Create a test markdown file
    test_file = Path(temp_dir) / "test.md"
    test_content = "# Test Markdown\n\nThis is a test."
    test_file.write_text(test_content)
    
    # Load the document
    doc = document_loader.load_document(str(test_file))
    
    assert isinstance(doc, Document)
    assert doc.content == test_content
    assert doc.metadata['file_name'] == "test.md"
    assert doc.metadata['file_type'] == ".md"


def test_load_html_file(document_loader, temp_dir):
    # Create a test HTML file
    test_file = Path(temp_dir) / "test.html"
    test_content = "<html><body><h1>Test</h1><p>This is a test.</p></body></html>"
    test_file.write_text(test_content)
    
    # Load the document
    doc = document_loader.load_document(str(test_file))
    
    assert isinstance(doc, Document)
    assert "Test" in doc.content
    assert "This is a test." in doc.content
    assert doc.metadata['file_name'] == "test.html"
    assert doc.metadata['file_type'] == ".html"


def test_load_documents_from_directory(document_loader, temp_dir):
    # Create multiple test files
    files = [
        ("doc1.txt", "Document 1 content"),
        ("doc2.md", "# Document 2\n\nContent"),
        ("doc3.txt", "Document 3 content")
    ]
    
    for filename, content in files:
        file_path = Path(temp_dir) / filename
        file_path.write_text(content)
    
    # Load all documents
    docs = document_loader.load_documents(temp_dir)
    
    assert len(docs) == 3
    assert all(isinstance(doc, Document) for doc in docs)
    
    # Check that all files were loaded
    loaded_files = {doc.metadata['file_name'] for doc in docs}
    expected_files = {"doc1.txt", "doc2.md", "doc3.txt"}
    assert loaded_files == expected_files


def test_unsupported_file_type(document_loader, temp_dir):
    # Create an unsupported file
    test_file = Path(temp_dir) / "test.xyz"
    test_file.write_text("Unsupported content")
    
    # Try to load the document
    with pytest.raises(ValueError, match="Unsupported file type"):
        document_loader.load_document(str(test_file))


def test_nonexistent_file(document_loader):
    with pytest.raises(ValueError, match="does not exist"):
        document_loader.load_document("/nonexistent/file.txt")


def test_nonexistent_directory(document_loader):
    with pytest.raises(ValueError, match="does not exist"):
        document_loader.load_documents("/nonexistent/directory")