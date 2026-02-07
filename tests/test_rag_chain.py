"""
Tests for RAG Chain Module

Tests for RAGChain and RAGResponse.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.rag_chain import RAGChain, RAGResponse, create_rag_chain
from src.chunker import Chunk
from src.vector_store import SearchResult
from src.llm_service import LLMResponse


class TestRAGResponse:
    """Tests for RAGResponse dataclass."""
    
    def test_response_creation(self):
        """Test basic response creation."""
        response = RAGResponse(
            answer="Test answer",
            sources=[{"source": "doc.pdf"}],
            retrieved_chunks=["chunk1"],
            confidence=0.85,
            query="Test query",
        )
        
        assert response.answer == "Test answer"
        assert len(response.sources) == 1
        assert response.confidence == 0.85
        assert response.query == "Test query"
    
    def test_response_to_dict(self):
        """Test serialization to dictionary."""
        response = RAGResponse(
            answer="Answer",
            sources=[{"source": "test.pdf", "score": 0.9}],
            retrieved_chunks=["chunk"],
            confidence=0.8,
            query="Question",
            metadata={"time": 1.5},
        )
        
        d = response.to_dict()
        
        assert d["answer"] == "Answer"
        assert d["sources"][0]["source"] == "test.pdf"
        assert d["confidence"] == 0.8
        assert d["metadata"]["time"] == 1.5
    
    def test_format_with_sources(self):
        """Test formatting answer with sources."""
        response = RAGResponse(
            answer="The bootcamp is 11 weeks long.",
            sources=[
                {"source": "schedule.pdf"},
                {"source": "faq.docx"},
            ],
            retrieved_chunks=[],
            confidence=0.9,
            query="How long is the bootcamp?",
        )
        
        formatted = response.format_with_sources()
        
        assert "The bootcamp is 11 weeks long." in formatted
        assert "Sources" in formatted
        assert "schedule.pdf" in formatted
        assert "faq.docx" in formatted
    
    def test_format_without_sources(self):
        """Test formatting when no sources."""
        response = RAGResponse(
            answer="No info found",
            sources=[],
            retrieved_chunks=[],
            confidence=0.0,
            query="Unknown topic",
        )
        
        formatted = response.format_with_sources()
        
        assert formatted == "No info found"


class TestRAGChain:
    """Tests for RAGChain class."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = Mock()
        store.search.return_value = []
        return store
    
    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        llm = Mock()
        llm.generate_with_context.return_value = LLMResponse(
            content="Generated response",
            model="test-model",
        )
        return llm
    
    def test_initialization(self, mock_vector_store, mock_llm_service):
        """Test RAG chain initialization."""
        chain = RAGChain(
            vector_store=mock_vector_store,
            llm_service=mock_llm_service,
        )
        
        assert chain.vector_store is mock_vector_store
        assert chain.llm_service is mock_llm_service
        assert chain.top_k > 0
        assert chain.similarity_threshold > 0
    
    def test_preprocess_query(self, mock_vector_store, mock_llm_service):
        """Test query preprocessing."""
        chain = RAGChain(
            vector_store=mock_vector_store,
            llm_service=mock_llm_service,
        )
        
        result = chain._preprocess_query("  What is AI?  ")
        
        assert result == "What is AI?"
    
    def test_build_context_empty(self, mock_vector_store, mock_llm_service):
        """Test context building with no results."""
        chain = RAGChain(
            vector_store=mock_vector_store,
            llm_service=mock_llm_service,
        )
        
        context = chain._build_context([])
        
        assert context == ""
    
    def test_build_context_with_results(self, mock_vector_store, mock_llm_service):
        """Test context building with search results."""
        chain = RAGChain(
            vector_store=mock_vector_store,
            llm_service=mock_llm_service,
        )
        
        # Create mock search results
        chunk = Chunk(text="AI is cool", source="doc.pdf", chunk_id="test_001", chunk_index=0)
        results = [SearchResult(chunk=chunk, score=0.9)]
        
        context = chain._build_context(results)
        
        assert "AI is cool" in context
        assert "doc.pdf" in context
    
    def test_calculate_confidence_no_results(self, mock_vector_store, mock_llm_service):
        """Test confidence calculation with no results."""
        chain = RAGChain(
            vector_store=mock_vector_store,
            llm_service=mock_llm_service,
        )
        
        confidence = chain._calculate_confidence([])
        
        assert confidence == 0.0
    
    def test_calculate_confidence_with_results(self, mock_vector_store, mock_llm_service):
        """Test confidence calculation with results."""
        chain = RAGChain(
            vector_store=mock_vector_store,
            llm_service=mock_llm_service,
        )
        
        chunk = Chunk(text="Text", source="doc.pdf", chunk_id="test_001", chunk_index=0)
        results = [
            SearchResult(chunk=chunk, score=0.9),
            SearchResult(chunk=chunk, score=0.8),
        ]
        
        confidence = chain._calculate_confidence(results)
        
        assert 0 < confidence <= 1.0
    
    def test_extract_sources(self, mock_vector_store, mock_llm_service):
        """Test source extraction from results."""
        chain = RAGChain(
            vector_store=mock_vector_store,
            llm_service=mock_llm_service,
        )
        
        chunk1 = Chunk(text="Text 1", source="doc1.pdf", chunk_id="test_001", chunk_index=0)
        chunk2 = Chunk(text="Text 2", source="doc2.pdf", chunk_id="test_002", chunk_index=0)
        results = [
            SearchResult(chunk=chunk1, score=0.9),
            SearchResult(chunk=chunk2, score=0.8),
        ]
        
        sources = chain._extract_sources(results)
        
        assert len(sources) == 2
        assert sources[0]["source"] == "doc1.pdf"
        assert sources[1]["source"] == "doc2.pdf"
    
    def test_query_no_results(self, mock_vector_store, mock_llm_service):
        """Test query when no relevant results found."""
        mock_vector_store.search.return_value = []
        
        chain = RAGChain(
            vector_store=mock_vector_store,
            llm_service=mock_llm_service,
        )
        
        response = chain.query("Unknown topic")
        
        assert response.confidence == 0.0
        assert "couldn't find" in response.answer.lower() or "don't have" in response.answer.lower()
    
    def test_query_with_results(self, mock_vector_store, mock_llm_service):
        """Test successful query with results."""
        chunk = Chunk(
            text="The bootcamp is 11 weeks long.",
            source="schedule.pdf",
            chunk_id="test_001",
            chunk_index=0,
        )
        mock_vector_store.search.return_value = [
            SearchResult(chunk=chunk, score=0.9)
        ]
        
        chain = RAGChain(
            vector_store=mock_vector_store,
            llm_service=mock_llm_service,
        )
        
        response = chain.query("How long is the bootcamp?")
        
        assert response.answer == "Generated response"
        assert len(response.sources) == 1
        assert response.confidence > 0
        mock_llm_service.generate_with_context.assert_called_once()
    
    def test_query_simple(self, mock_vector_store, mock_llm_service):
        """Test simple query returning just string."""
        chunk = Chunk(text="Test", source="doc.pdf", chunk_id="test_001", chunk_index=0)
        mock_vector_store.search.return_value = [
            SearchResult(chunk=chunk, score=0.9)
        ]
        
        chain = RAGChain(
            vector_store=mock_vector_store,
            llm_service=mock_llm_service,
        )
        
        answer = chain.query_simple("Question?")
        
        assert answer == "Generated response"
    
    def test_get_relevant_chunks(self, mock_vector_store, mock_llm_service):
        """Test retrieving chunks without generation."""
        chunk = Chunk(text="Relevant text", source="doc.pdf", chunk_id="test_001", chunk_index=0)
        mock_vector_store.search.return_value = [
            SearchResult(chunk=chunk, score=0.85)
        ]
        
        chain = RAGChain(
            vector_store=mock_vector_store,
            llm_service=mock_llm_service,
        )
        
        results = chain.get_relevant_chunks("Query")
        
        assert len(results) == 1
        assert results[0].chunk.text == "Relevant text"
        # LLM should not be called
        mock_llm_service.generate_with_context.assert_not_called()


class TestCreateRAGChain:
    """Tests for the factory function."""
    
    @patch('src.rag_chain.EmbeddingService')
    @patch('src.rag_chain.VectorStore')
    @patch('src.rag_chain.LLMService')
    def test_create_rag_chain(
        self, 
        mock_llm_cls, 
        mock_vector_cls, 
        mock_embed_cls
    ):
        """Test factory function creates all components."""
        mock_embed = Mock()
        mock_vector = Mock()
        mock_llm = Mock()
        
        mock_embed_cls.return_value = mock_embed
        mock_vector_cls.return_value = mock_vector
        mock_llm_cls.return_value = mock_llm
        
        chain = create_rag_chain(
            embedding_provider="local",
            llm_provider="ollama",
            vector_store_provider="faiss",
        )
        
        assert isinstance(chain, RAGChain)
        mock_embed_cls.assert_called_once_with(provider="local")
        mock_llm_cls.assert_called_once_with(provider="ollama")
