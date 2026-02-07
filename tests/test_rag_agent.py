"""
Tests for RAG Agent Module

Tests for RAGAgent - the public API.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from src.rag_agent import RAGAgent, create_agent
from src.rag_chain import RAGResponse


class TestRAGAgent:
    """Tests for RAGAgent class."""
    
    @patch('src.rag_agent.EmbeddingService')
    @patch('src.rag_agent.VectorStore')
    @patch('src.rag_agent.LLMService')
    @patch('src.rag_agent.RAGChain')
    def test_initialization(
        self,
        mock_chain_cls,
        mock_llm_cls,
        mock_vector_cls,
        mock_embed_cls,
    ):
        """Test agent initialization."""
        agent = RAGAgent(
            embedding_provider="local",
            llm_provider="ollama",
            vector_store_provider="faiss",
        )
        
        mock_embed_cls.assert_called_once()
        mock_vector_cls.assert_called_once()
        mock_llm_cls.assert_called_once()
        mock_chain_cls.assert_called_once()
    
    @patch('src.rag_agent.EmbeddingService')
    @patch('src.rag_agent.VectorStore')
    @patch('src.rag_agent.LLMService')
    @patch('src.rag_agent.RAGChain')
    def test_query_returns_dict(
        self,
        mock_chain_cls,
        mock_llm_cls,
        mock_vector_cls,
        mock_embed_cls,
    ):
        """Test that query returns API contract format."""
        # Setup mock chain
        mock_chain = Mock()
        mock_chain.query.return_value = RAGResponse(
            answer="Test answer",
            sources=[{"source": "test.pdf"}],
            retrieved_chunks=["chunk1"],
            confidence=0.9,
            query="Test question",
        )
        mock_chain_cls.return_value = mock_chain
        
        agent = RAGAgent()
        result = agent.query("Test question")
        
        # Verify API contract
        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result
        assert "retrieved_chunks" in result
        
        assert result["answer"] == "Test answer"
        assert result["confidence"] == 0.9
    
    @patch('src.rag_agent.EmbeddingService')
    @patch('src.rag_agent.VectorStore')
    @patch('src.rag_agent.LLMService')
    @patch('src.rag_agent.RAGChain')
    def test_query_with_conversation_id(
        self,
        mock_chain_cls,
        mock_llm_cls,
        mock_vector_cls,
        mock_embed_cls,
    ):
        """Test query with conversation memory."""
        mock_chain = Mock()
        mock_chain.query.return_value = RAGResponse(
            answer="Contextual answer",
            sources=[],
            retrieved_chunks=[],
            confidence=0.8,
            query="Follow up",
        )
        mock_chain_cls.return_value = mock_chain
        
        agent = RAGAgent()
        
        # First query
        result1 = agent.query("What is AI?", conversation_id="user_123")
        # Follow up
        result2 = agent.query("Tell me more", conversation_id="user_123")
        
        assert result2["answer"] == "Contextual answer"
    
    @patch('src.rag_agent.EmbeddingService')
    @patch('src.rag_agent.VectorStore')
    @patch('src.rag_agent.LLMService')
    @patch('src.rag_agent.RAGChain')
    def test_query_error_handling(
        self,
        mock_chain_cls,
        mock_llm_cls,
        mock_vector_cls,
        mock_embed_cls,
    ):
        """Test error handling in query."""
        mock_chain = Mock()
        mock_chain.query.side_effect = Exception("LLM error")
        mock_chain_cls.return_value = mock_chain
        
        agent = RAGAgent()
        result = agent.query("Test")
        
        # Should return graceful error response
        assert "error" in result or "sorry" in result["answer"].lower()
        assert result["confidence"] == 0.0
    
    @patch('src.rag_agent.EmbeddingService')
    @patch('src.rag_agent.VectorStore')
    @patch('src.rag_agent.LLMService')
    @patch('src.rag_agent.RAGChain')
    @patch('src.rag_agent.DocumentChunker')
    def test_ingest_document(
        self,
        mock_chunker_cls,
        mock_chain_cls,
        mock_llm_cls,
        mock_vector_cls,
        mock_embed_cls,
        tmp_path,
    ):
        """Test document ingestion."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        # Setup mocks
        mock_chunker = Mock()
        mock_chunker.process_document.return_value = [Mock()]
        mock_chunker_cls.return_value = mock_chunker
        
        mock_vector = Mock()
        mock_vector.add_chunks.return_value = 1
        mock_vector_cls.return_value = mock_vector
        
        agent = RAGAgent()
        result = agent.ingest_document(str(test_file))
        
        assert result is True
        mock_chunker.process_document.assert_called_once()
    
    @patch('src.rag_agent.EmbeddingService')
    @patch('src.rag_agent.VectorStore')
    @patch('src.rag_agent.LLMService')
    @patch('src.rag_agent.RAGChain')
    def test_ingest_nonexistent_file(
        self,
        mock_chain_cls,
        mock_llm_cls,
        mock_vector_cls,
        mock_embed_cls,
    ):
        """Test ingestion of nonexistent file."""
        agent = RAGAgent()
        result = agent.ingest_document("/nonexistent/path.pdf")
        
        assert result is False
    
    @patch('src.rag_agent.EmbeddingService')
    @patch('src.rag_agent.VectorStore')
    @patch('src.rag_agent.LLMService')
    @patch('src.rag_agent.RAGChain')
    @patch('src.rag_agent.DocumentChunker')
    def test_ingest_text(
        self,
        mock_chunker_cls,
        mock_chain_cls,
        mock_llm_cls,
        mock_vector_cls,
        mock_embed_cls,
    ):
        """Test direct text ingestion."""
        mock_chunker = Mock()
        mock_chunker.process_text.return_value = [Mock()]
        mock_chunker_cls.return_value = mock_chunker
        
        mock_vector = Mock()
        mock_vector.add_chunks.return_value = 1
        mock_vector_cls.return_value = mock_vector
        
        agent = RAGAgent()
        result = agent.ingest_text("Some text content", source_name="manual_input")
        
        assert result is True
        mock_chunker.process_text.assert_called_once()
    
    @patch('src.rag_agent.EmbeddingService')
    @patch('src.rag_agent.VectorStore')
    @patch('src.rag_agent.LLMService')
    @patch('src.rag_agent.RAGChain')
    def test_clear_knowledge_base(
        self,
        mock_chain_cls,
        mock_llm_cls,
        mock_vector_cls,
        mock_embed_cls,
    ):
        """Test clearing knowledge base."""
        mock_vector = Mock()
        mock_vector_cls.return_value = mock_vector
        
        agent = RAGAgent()
        result = agent.clear_knowledge_base()
        
        assert result is True
        mock_vector.clear.assert_called_once()
    
    @patch('src.rag_agent.EmbeddingService')
    @patch('src.rag_agent.VectorStore')
    @patch('src.rag_agent.LLMService')
    @patch('src.rag_agent.RAGChain')
    def test_get_stats(
        self,
        mock_chain_cls,
        mock_llm_cls,
        mock_vector_cls,
        mock_embed_cls,
    ):
        """Test getting system stats."""
        mock_embed = Mock()
        mock_embed.provider_name = "local"
        mock_embed.model_name = "all-MiniLM-L6-v2"
        mock_embed.dimension = 384
        mock_embed_cls.return_value = mock_embed
        
        mock_vector = Mock()
        mock_vector.provider = "faiss"
        mock_vector.count.return_value = 100
        mock_vector_cls.return_value = mock_vector
        
        mock_llm = Mock()
        mock_llm.provider_name = "ollama"
        mock_llm.model_name = "llama2"
        mock_llm_cls.return_value = mock_llm
        
        agent = RAGAgent()
        stats = agent.get_stats()
        
        assert "knowledge_base" in stats
        assert "embedding" in stats
        assert "llm" in stats
        assert stats["knowledge_base"]["total_chunks"] == 100
    
    @patch('src.rag_agent.EmbeddingService')
    @patch('src.rag_agent.VectorStore')
    @patch('src.rag_agent.LLMService')
    @patch('src.rag_agent.RAGChain')
    def test_clear_conversation(
        self,
        mock_chain_cls,
        mock_llm_cls,
        mock_vector_cls,
        mock_embed_cls,
    ):
        """Test clearing specific conversation."""
        agent = RAGAgent()
        
        # Create a conversation first
        agent.query("Hello", conversation_id="test_user")
        
        # Clear it
        result = agent.clear_conversation("test_user")
        
        assert result is True
    
    @patch('src.rag_agent.EmbeddingService')
    @patch('src.rag_agent.VectorStore')
    @patch('src.rag_agent.LLMService')
    @patch('src.rag_agent.RAGChain')
    def test_search_similar(
        self,
        mock_chain_cls,
        mock_llm_cls,
        mock_vector_cls,
        mock_embed_cls,
    ):
        """Test similarity search."""
        from src.chunker import Chunk
        from src.vector_store import SearchResult
        
        chunk = Chunk(text="Test content", source="doc.pdf", chunk_id="test_001", chunk_index=0)
        mock_chain = Mock()
        mock_chain.get_relevant_chunks.return_value = [
            SearchResult(chunk=chunk, score=0.9)
        ]
        mock_chain_cls.return_value = mock_chain
        
        agent = RAGAgent()
        results = agent.search_similar("query", top_k=3)
        
        assert len(results) == 1
        assert results[0]["text"] == "Test content"
        assert results[0]["score"] == 0.9


class TestCreateAgent:
    """Tests for create_agent factory function."""
    
    @patch('src.rag_agent.RAGAgent')
    def test_create_agent_default(self, mock_agent_cls):
        """Test factory with defaults."""
        mock_agent_cls.return_value = Mock()
        
        agent = create_agent(auto_load=False)
        
        mock_agent_cls.assert_called_once_with(auto_load_knowledge_base=False)
    
    @patch('src.rag_agent.RAGAgent')
    def test_create_agent_with_kwargs(self, mock_agent_cls):
        """Test factory with additional kwargs."""
        mock_agent_cls.return_value = Mock()
        
        agent = create_agent(
            auto_load=True,
            llm_provider="openai",
        )
        
        mock_agent_cls.assert_called_once_with(
            auto_load_knowledge_base=True,
            llm_provider="openai",
        )
