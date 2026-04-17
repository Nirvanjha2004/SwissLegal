"""Tests for chunker module functionality."""

from src.chunker import TextChunk, chunk_text, chunk_records


def test_chunk_records_basic_functionality():
    """Test basic chunk_records functionality with simple input."""
    records = [
        ("doc1", "This is the first document with some text content."),
        ("doc2", "This is the second document with different content."),
    ]
    
    chunks = chunk_records(records, chunk_size=20, overlap=5)
    
    # Verify we get chunks
    assert len(chunks) > 0
    
    # Verify sequential chunk IDs starting from 0
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    assert chunk_ids == list(range(len(chunks)))
    
    # Verify source traceability
    assert all(chunk.source_id in ["doc1", "doc2"] for chunk in chunks)
    
    # Verify all chunks have non-empty text
    assert all(chunk.text.strip() for chunk in chunks)


def test_chunk_records_empty_documents():
    """Test that empty documents are skipped."""
    records = [
        ("doc1", "Some content here"),
        ("doc2", ""),  # Empty document
        ("doc3", "   "),  # Whitespace only
        ("doc4", "More content"),
    ]
    
    chunks = chunk_records(records, chunk_size=50, overlap=0)
    
    # Should only have chunks from doc1 and doc4
    source_ids = [chunk.source_id for chunk in chunks]
    assert "doc1" in source_ids
    assert "doc4" in source_ids
    assert all(chunk.text.strip() for chunk in chunks)


def test_chunk_records_sequential_ids():
    """Test that chunk IDs are assigned sequentially across all documents."""
    records = [
        ("doc1", "First document with enough content to create multiple chunks when using small chunk size."),
        ("doc2", "Second document also with enough content to create multiple chunks when using small chunk size."),
    ]
    
    chunks = chunk_records(records, chunk_size=20, overlap=5)
    
    # Verify sequential IDs
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == i


def test_chunk_records_empty_input():
    """Test chunk_records with empty input."""
    chunks = chunk_records([], chunk_size=100, overlap=10)
    assert chunks == []


if __name__ == "__main__":
    # Run basic tests
    test_chunk_records_basic_functionality()
    test_chunk_records_empty_documents()
    test_chunk_records_sequential_ids()
    test_chunk_records_empty_input()
    print("All tests passed!")