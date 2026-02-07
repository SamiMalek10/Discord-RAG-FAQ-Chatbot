"""
Tests for EmbeddingService module.

Run with: pytest tests/test_embeddings.py -v

Note: Some tests require sentence-transformers installed.
"""

import pytest
import numpy as np

from src.embeddings import (
    EmbeddingService,
    LocalEmbeddingProvider,
    cosine_similarity,
    compare_embeddings,
)


class TestCosineSimiliarity:
    """Tests for cosine similarity function."""
    
    def test_identical_vectors(self):
        """Identical vectors should have similarity 1."""
        vec = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(vec, vec) - 1.0) < 0.001
    
    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(vec1, vec2)) < 0.001
    
    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        assert abs(cosine_similarity(vec1, vec2) + 1.0) < 0.001
    
    def test_zero_vector(self):
        """Zero vector should return 0 similarity."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [0.0, 0.0, 0.0]
        assert cosine_similarity(vec1, vec2) == 0.0


class TestLocalEmbeddingProvider:
    """Tests for local embedding provider."""
    
    @pytest.fixture
    def provider(self):
        """Create a local embedding provider."""
        return LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    
    @pytest.mark.slow
    def test_embed_text(self, provider):
        """Test single text embedding."""
        embedding = provider.embed_text("Hello, world!")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384  # MiniLM dimension
        assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.slow
    def test_embed_batch(self, provider):
        """Test batch embedding."""
        texts = ["First text", "Second text", "Third text"]
        embeddings = provider.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)
    
    @pytest.mark.slow
    def test_dimension_property(self, provider):
        """Test dimension property."""
        _ = provider.embed_text("test")  # Trigger model load
        assert provider.dimension == 384
    
    @pytest.mark.slow
    def test_similar_texts_have_high_similarity(self, provider):
        """Test that similar texts have high cosine similarity."""
        text1 = "The cat sat on the mat."
        text2 = "A cat is sitting on a mat."
        text3 = "The weather is sunny today."
        
        emb1 = provider.embed_text(text1)
        emb2 = provider.embed_text(text2)
        emb3 = provider.embed_text(text3)
        
        sim_similar = cosine_similarity(emb1, emb2)
        sim_different = cosine_similarity(emb1, emb3)
        
        # Similar texts should have higher similarity
        assert sim_similar > sim_different
        assert sim_similar > 0.5  # Should be reasonably high


class TestEmbeddingService:
    """Tests for the main EmbeddingService class."""
    
    @pytest.fixture
    def service(self):
        """Create an embedding service with local provider."""
        return EmbeddingService(provider="local")
    
    @pytest.mark.slow
    def test_service_initialization(self, service):
        """Test service initializes correctly."""
        assert service.provider_name == "local"
        assert service.model_name == "all-MiniLM-L6-v2"
    
    @pytest.mark.slow
    def test_embed_text(self, service):
        """Test text embedding through service."""
        embedding = service.embed_text("Test embedding")
        
        assert isinstance(embedding, list)
        assert len(embedding) == service.dimension
    
    @pytest.mark.slow
    def test_embed_query(self, service):
        """Test query embedding (alias for embed_text)."""
        query = "What is the schedule?"
        embedding = service.embed_query(query)
        
        assert len(embedding) == service.dimension
    
    @pytest.mark.slow
    def test_embed_batch(self, service):
        """Test batch embedding."""
        texts = ["Text one", "Text two"]
        embeddings = service.embed_batch(texts)
        
        assert len(embeddings) == 2
    
    def test_empty_text_raises_error(self, service):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            service.embed_text("")
    
    @pytest.mark.slow
    def test_embed_batch_filters_empty(self, service):
        """Test that batch embedding filters empty texts."""
        texts = ["Valid text", "", "   ", "Another valid"]
        embeddings = service.embed_batch(texts)
        
        # Should only return embeddings for non-empty texts
        assert len(embeddings) == 2


class TestCompareEmbeddings:
    """Tests for the compare_embeddings utility."""
    
    @pytest.mark.slow
    def test_compare_identical_texts(self):
        """Identical texts should have very high similarity."""
        text = "This is a test sentence."
        similarity = compare_embeddings(text, text)
        
        assert similarity > 0.99
    
    @pytest.mark.slow
    def test_compare_similar_texts(self):
        """Similar texts should have reasonable similarity."""
        text1 = "The quick brown fox jumps over the lazy dog."
        text2 = "A fast brown fox leaps over a sleepy dog."
        
        similarity = compare_embeddings(text1, text2)
        
        assert similarity > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
