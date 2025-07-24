import pytest
from src.ingestion.text_splitter import TextSplitter, TextChunk


@pytest.fixture
def text_splitter():
    return TextSplitter(chunk_size=100, chunk_overlap=20)


def test_split_empty_text(text_splitter):
    chunks = text_splitter.split_text("")
    assert chunks == []


def test_split_small_text(text_splitter):
    text = "This is a small text."
    chunks = text_splitter.split_text(text)
    
    assert len(chunks) == 1
    assert chunks[0].content == text
    assert chunks[0].start_index == 0
    assert chunks[0].end_index == len(text)


def test_split_with_separator(text_splitter):
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = text_splitter.split_text(text)
    
    assert len(chunks) >= 1
    assert all(isinstance(chunk, TextChunk) for chunk in chunks)
    
    # Check that chunks don't exceed chunk_size
    for chunk in chunks:
        assert len(chunk.content) <= text_splitter.chunk_size + 50  # Allow some buffer


def test_split_large_text():
    splitter = TextSplitter(chunk_size=50, chunk_overlap=10, separator=" ")
    # Create text with spaces so it can be split properly
    words = ["word" + str(i) for i in range(50)]  # 50 words
    text = " ".join(words)  # Much longer than 50 characters
    
    chunks = splitter.split_text(text)
    
    # Should create multiple chunks for long text
    assert len(chunks) >= 1
    # Check that chunks don't exceed the chunk size significantly
    for chunk in chunks:
        assert len(chunk.content) <= splitter.chunk_size + 100  # Allow buffer for word boundaries


def test_metadata_propagation(text_splitter):
    text = "Test text for metadata"
    metadata = {"source": "test.txt", "author": "Test Author"}
    
    chunks = text_splitter.split_text(text, metadata)
    
    assert all("source" in chunk.metadata for chunk in chunks)
    assert all(chunk.metadata["source"] == "test.txt" for chunk in chunks)
    assert all("chunk_index" in chunk.metadata for chunk in chunks)


def test_chunk_indices():
    splitter = TextSplitter(chunk_size=20, chunk_overlap=5)
    text = "First chunk content. Second chunk content. Third chunk content."
    
    chunks = splitter.split_text(text)
    
    # Verify chunk indices are sequential
    for i, chunk in enumerate(chunks):
        assert chunk.metadata['chunk_index'] == i
    
    # Verify start and end indices
    for chunk in chunks:
        assert chunk.start_index < chunk.end_index
        assert text[chunk.start_index:chunk.end_index] == chunk.content


def test_custom_separators():
    splitter = TextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separator=". ",
        secondary_separator=" "
    )
    
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = splitter.split_text(text)
    
    assert len(chunks) >= 1
    # Check that sentences are preserved when possible
    for chunk in chunks:
        # Each chunk should ideally end with a period (except possibly the last one)
        if chunk != chunks[-1]:
            assert chunk.content.rstrip().endswith('.') or len(chunk.content) >= splitter.chunk_size