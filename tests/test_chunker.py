"""
Tests for DocumentChunker module.

Run with: pytest tests/test_chunker.py -v
"""

import pytest
from pathlib import Path
import tempfile
import os

from src.chunker import DocumentChunker, Chunk, chunk_document


class TestChunk:
    """Tests for Chunk dataclass."""
    
    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = Chunk(
            text="This is a test chunk.",
            chunk_id="test_0_abc123",
            source="test.pdf",
            chunk_index=0,
            total_chunks=5,
        )
        
        assert chunk.text == "This is a test chunk."
        assert chunk.source == "test.pdf"
        assert chunk.chunk_index == 0
        assert chunk.total_chunks == 5
    
    def test_chunk_auto_id(self):
        """Test automatic chunk ID generation."""
        chunk = Chunk(
            text="Test content for ID generation.",
            chunk_id="",
            source="document.pdf",
            chunk_index=3,
        )
        
        # ID should be generated
        assert chunk.chunk_id != ""
        assert "document" in chunk.chunk_id
        assert "3" in chunk.chunk_id
    
    def test_chunk_to_dict(self):
        """Test chunk serialization."""
        chunk = Chunk(
            text="Test text",
            chunk_id="test_id",
            source="source.txt",
            chunk_index=0,
            metadata={"page": 1},
        )
        
        data = chunk.to_dict()
        
        assert data["text"] == "Test text"
        assert data["chunk_id"] == "test_id"
        assert data["metadata"]["page"] == 1
    
    def test_chunk_from_dict(self):
        """Test chunk deserialization."""
        data = {
            "text": "Restored text",
            "chunk_id": "restored_id",
            "source": "restored.pdf",
            "chunk_index": 2,
            "total_chunks": 10,
            "metadata": {"section": "intro"},
        }
        
        chunk = Chunk.from_dict(data)
        
        assert chunk.text == "Restored text"
        assert chunk.chunk_id == "restored_id"
        assert chunk.metadata["section"] == "intro"


class TestDocumentChunker:
    """Tests for DocumentChunker class."""
    
    @pytest.fixture
    def chunker(self):
        """Create a chunker with small chunk size for testing."""
        return DocumentChunker(chunk_size=100, chunk_overlap=20)
    
    @pytest.fixture
    def sample_text_file(self, tmp_path):
        """Create a temporary text file for testing."""
        content = """# Introduction

This is the first paragraph of the document. It contains some text that will be chunked.

## Section 1

This is the content of section 1. It has multiple sentences. Each sentence provides information.

## Section 2

This is the content of section 2. More content here. Additional sentences for testing.

## Conclusion

This is the conclusion of the document. Final thoughts are written here.
"""
        file_path = tmp_path / "test_document.txt"
        file_path.write_text(content, encoding="utf-8")
        return file_path
    
    def test_chunker_initialization(self, chunker):
        """Test chunker initialization."""
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 20
    
    def test_process_text_file(self, chunker, sample_text_file):
        """Test processing a text file."""
        chunks = chunker.process_document(sample_text_file)
        
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.source == sample_text_file.name for c in chunks)
    
    def test_chunks_have_sequential_indices(self, chunker, sample_text_file):
        """Test that chunks have correct sequential indices."""
        chunks = chunker.process_document(sample_text_file)
        
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))
    
    def test_total_chunks_set_correctly(self, chunker, sample_text_file):
        """Test that total_chunks is set correctly."""
        chunks = chunker.process_document(sample_text_file)
        
        total = len(chunks)
        assert all(c.total_chunks == total for c in chunks)
    
    def test_process_text_directly(self, chunker):
        """Test processing raw text."""
        text = """
        This is a long text that needs to be chunked into smaller pieces.
        Each piece should maintain context and be properly indexed.
        The chunker should handle this appropriately.
        """
        
        chunks = chunker.process_text(text, source_name="direct_test")
        
        assert len(chunks) > 0
        assert all(c.source == "direct_test" for c in chunks)
    
    def test_additional_metadata(self, chunker, sample_text_file):
        """Test that additional metadata is preserved."""
        metadata = {"category": "FAQ", "language": "en"}
        
        chunks = chunker.process_document(
            sample_text_file,
            additional_metadata=metadata,
        )
        
        for chunk in chunks:
            assert chunk.metadata["category"] == "FAQ"
            assert chunk.metadata["language"] == "en"
    
    def test_unsupported_file_type(self, chunker, tmp_path):
        """Test handling of unsupported file types."""
        file_path = tmp_path / "test.xyz"
        file_path.write_text("content")
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            chunker.process_document(file_path)
    
    def test_file_not_found(self, chunker):
        """Test handling of missing files."""
        with pytest.raises(FileNotFoundError):
            chunker.process_document("/nonexistent/file.txt")
    
    def test_empty_chunks_filtered(self, chunker):
        """Test that empty/short chunks are filtered out."""
        text = "Short\n\n\n\n\n\nAnother short one."
        
        chunks = chunker.process_text(text)
        
        # Very short chunks should be filtered
        for chunk in chunks:
            assert len(chunk.text.strip()) >= 10


class TestChunkDocument:
    """Tests for the convenience function."""
    
    def test_chunk_document_function(self, tmp_path):
        """Test the quick chunk_document function."""
        content = "This is test content. " * 20
        file_path = tmp_path / "quick_test.txt"
        file_path.write_text(content)
        
        chunks = chunk_document(file_path)
        
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
