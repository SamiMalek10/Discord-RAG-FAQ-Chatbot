"""
Tests for MongoDB Vector Store

Tests for MongoDBVectorStore implementation.
Uses mocking to avoid needing a real MongoDB connection.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vector_store import MongoDBVectorStore, SearchResult
from src.chunker import Chunk


class TestMongoDBVectorStore:
    """Tests for MongoDBVectorStore class."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock()
        settings.vector_store = Mock()
        settings.vector_store.mongodb_uri = "mongodb+srv://test:test@cluster.mongodb.net/"
        settings.vector_store.mongodb_database = "test_db"
        settings.vector_store.mongodb_collection = "test_collection"
        settings.vector_store.mongodb_vector_index = "test_index"
        return settings
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks with embeddings."""
        return [
            Chunk(
                text="The AI Bootcamp starts in Week 1",
                chunk_id="chunk_1",
                source="bootcamp.docx",
                chunk_index=0,
                embedding=[0.1, 0.2, 0.3] * 128  # 384 dimensions
            ),
            Chunk(
                text="Phase 2 involves building the RAG pipeline",
                chunk_id="chunk_2",
                source="bootcamp.docx",
                chunk_index=1,
                embedding=[0.2, 0.3, 0.4] * 128
            ),
            Chunk(
                text="The project deadline is in 3 weeks",
                chunk_id="chunk_3",
                source="faq.docx",
                chunk_index=0,
                embedding=[0.3, 0.4, 0.5] * 128
            ),
        ]
    
    @patch('src.vector_store.get_settings')
    def test_initialization(self, mock_get_settings, mock_settings):
        """Test MongoDBVectorStore initialization."""
        mock_get_settings.return_value = mock_settings
        
        store = MongoDBVectorStore(
            dimension=384,
            uri="mongodb+srv://test:test@cluster.mongodb.net/",
            database="test_db",
            collection="test_collection",
            vector_index="test_index"
        )
        
        assert store.dimension == 384
        assert store.database_name == "test_db"
        assert store.collection_name == "test_collection"
        assert store.vector_index == "test_index"
        assert store._collection is None  # Not connected yet
    
    @patch('src.vector_store.get_settings')
    @patch('pymongo.MongoClient')
    def test_connect(self, mock_mongo_client, mock_get_settings, mock_settings):
        """Test MongoDB connection."""
        mock_get_settings.return_value = mock_settings
        
        # Setup mock client
        mock_client = MagicMock()
        mock_mongo_client.return_value = mock_client
        
        store = MongoDBVectorStore(
            dimension=384,
            uri="mongodb+srv://test:test@cluster.mongodb.net/"
        )
        
        # Trigger connection
        store._connect()
        
        # Verify connection was made
        mock_mongo_client.assert_called_once()
        mock_client.admin.command.assert_called_with("ping")
    
    @patch('src.vector_store.get_settings')
    @patch('pymongo.MongoClient')
    def test_add_chunks(self, mock_mongo_client, mock_get_settings, mock_settings, sample_chunks):
        """Test adding chunks to MongoDB."""
        mock_get_settings.return_value = mock_settings
        
        # Setup mock client and collection
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.__getitem__.return_value.__getitem__.return_value = mock_collection
        mock_mongo_client.return_value = mock_client
        
        # Mock bulk_write result
        mock_result = MagicMock()
        mock_result.upserted_count = 2
        mock_result.modified_count = 1
        mock_collection.bulk_write.return_value = mock_result
        
        store = MongoDBVectorStore(
            dimension=384,
            uri="mongodb+srv://test:test@cluster.mongodb.net/"
        )
        
        added = store.add_chunks(sample_chunks)
        
        assert added == 3  # 2 upserted + 1 modified
        mock_collection.bulk_write.assert_called_once()
    
    @patch('src.vector_store.get_settings')
    @patch('pymongo.MongoClient')
    def test_add_empty_chunks(self, mock_mongo_client, mock_get_settings, mock_settings):
        """Test adding empty chunk list."""
        mock_get_settings.return_value = mock_settings
        
        store = MongoDBVectorStore(
            dimension=384,
            uri="mongodb+srv://test:test@cluster.mongodb.net/"
        )
        
        added = store.add_chunks([])
        assert added == 0
    
    @patch('src.vector_store.get_settings')
    @patch('pymongo.MongoClient')
    def test_add_chunks_without_embeddings(self, mock_mongo_client, mock_get_settings, mock_settings):
        """Test adding chunks without embeddings (should be skipped)."""
        mock_get_settings.return_value = mock_settings
        
        # Chunks without embeddings
        chunks = [
            Chunk(text="No embedding", chunk_id="c1", source="test.docx", chunk_index=0),
            Chunk(text="Also no embedding", chunk_id="c2", source="test.docx", chunk_index=1),
        ]
        
        # Setup mock
        mock_client = MagicMock()
        mock_mongo_client.return_value = mock_client
        
        store = MongoDBVectorStore(
            dimension=384,
            uri="mongodb+srv://test:test@cluster.mongodb.net/"
        )
        
        added = store.add_chunks(chunks)
        assert added == 0
    
    @patch('src.vector_store.get_settings')
    @patch('pymongo.MongoClient')
    def test_search(self, mock_mongo_client, mock_get_settings, mock_settings):
        """Test vector search."""
        mock_get_settings.return_value = mock_settings
        
        # Setup mock client and collection
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.__getitem__.return_value.__getitem__.return_value = mock_collection
        mock_mongo_client.return_value = mock_client
        
        # Mock aggregation results
        mock_collection.aggregate.return_value = iter([
            {
                "_id": "chunk_1",
                "text": "The AI Bootcamp starts in Week 1",
                "source": "bootcamp.docx",
                "chunk_index": 0,
                "total_chunks": 10,
                "metadata": {},
                "embedding": [0.1] * 384,
                "score": 0.95
            },
            {
                "_id": "chunk_2",
                "text": "Phase 2 involves building the RAG pipeline",
                "source": "bootcamp.docx",
                "chunk_index": 1,
                "total_chunks": 10,
                "metadata": {},
                "embedding": [0.2] * 384,
                "score": 0.85
            },
        ])
        
        store = MongoDBVectorStore(
            dimension=384,
            uri="mongodb+srv://test:test@cluster.mongodb.net/"
        )
        
        results = store.search(
            query_embedding=[0.1] * 384,
            top_k=5,
            threshold=0.5
        )
        
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].score == 0.95
        assert results[0].rank == 1
        assert results[1].score == 0.85
        assert results[1].rank == 2
    
    @patch('src.vector_store.get_settings')
    @patch('pymongo.MongoClient')
    def test_search_with_threshold(self, mock_mongo_client, mock_get_settings, mock_settings):
        """Test search with similarity threshold filtering."""
        mock_get_settings.return_value = mock_settings
        
        # Setup mock
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.__getitem__.return_value.__getitem__.return_value = mock_collection
        mock_mongo_client.return_value = mock_client
        
        # Mock results with varying scores
        mock_collection.aggregate.return_value = iter([
            {"_id": "c1", "text": "High score", "source": "a.docx", 
             "chunk_index": 0, "score": 0.9, "embedding": []},
            {"_id": "c2", "text": "Low score", "source": "a.docx", 
             "chunk_index": 1, "score": 0.3, "embedding": []},  # Below threshold
        ])
        
        store = MongoDBVectorStore(
            dimension=384,
            uri="mongodb+srv://test:test@cluster.mongodb.net/"
        )
        
        results = store.search(
            query_embedding=[0.1] * 384,
            top_k=5,
            threshold=0.5  # Should filter out 0.3 score
        )
        
        assert len(results) == 1
        assert results[0].score == 0.9
    
    @patch('src.vector_store.get_settings')
    @patch('pymongo.MongoClient')
    def test_delete(self, mock_mongo_client, mock_get_settings, mock_settings):
        """Test deleting chunks."""
        mock_get_settings.return_value = mock_settings
        
        # Setup mock
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.__getitem__.return_value.__getitem__.return_value = mock_collection
        mock_mongo_client.return_value = mock_client
        
        # Mock delete result
        mock_result = MagicMock()
        mock_result.deleted_count = 2
        mock_collection.delete_many.return_value = mock_result
        
        store = MongoDBVectorStore(
            dimension=384,
            uri="mongodb+srv://test:test@cluster.mongodb.net/"
        )
        
        deleted = store.delete(["chunk_1", "chunk_2"])
        
        assert deleted == 2
        mock_collection.delete_many.assert_called_once()
    
    @patch('src.vector_store.get_settings')
    @patch('pymongo.MongoClient')
    def test_clear(self, mock_mongo_client, mock_get_settings, mock_settings):
        """Test clearing all documents."""
        mock_get_settings.return_value = mock_settings
        
        # Setup mock
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.__getitem__.return_value.__getitem__.return_value = mock_collection
        mock_mongo_client.return_value = mock_client
        
        # Mock delete result
        mock_result = MagicMock()
        mock_result.deleted_count = 100
        mock_collection.delete_many.return_value = mock_result
        
        store = MongoDBVectorStore(
            dimension=384,
            uri="mongodb+srv://test:test@cluster.mongodb.net/"
        )
        
        store.clear()
        
        mock_collection.delete_many.assert_called_with({})
    
    @patch('src.vector_store.get_settings')
    @patch('pymongo.MongoClient')
    def test_count(self, mock_mongo_client, mock_get_settings, mock_settings):
        """Test counting documents."""
        mock_get_settings.return_value = mock_settings
        
        # Setup mock
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.__getitem__.return_value.__getitem__.return_value = mock_collection
        mock_mongo_client.return_value = mock_client
        
        mock_collection.count_documents.return_value = 150
        
        store = MongoDBVectorStore(
            dimension=384,
            uri="mongodb+srv://test:test@cluster.mongodb.net/"
        )
        
        count = store.count()
        
        assert count == 150
        mock_collection.count_documents.assert_called_with({})
    
    @patch('src.vector_store.get_settings')
    def test_missing_uri_raises_error(self, mock_get_settings, mock_settings):
        """Test that missing URI raises ValueError on connect."""
        mock_settings.vector_store.mongodb_uri = None
        mock_get_settings.return_value = mock_settings
        
        store = MongoDBVectorStore(dimension=384)
        
        with pytest.raises(ValueError, match="MongoDB URI not configured"):
            store._connect()
    
    @patch('src.vector_store.get_settings')
    def test_vector_index_definition(self, mock_get_settings, mock_settings):
        """Test vector index definition generation."""
        mock_get_settings.return_value = mock_settings
        
        store = MongoDBVectorStore(
            dimension=384,
            uri="mongodb+srv://test:test@cluster.mongodb.net/"
        )
        
        index_def = store.create_vector_index()
        
        assert "mappings" in index_def
        assert "fields" in index_def["mappings"]
        assert "embedding" in index_def["mappings"]["fields"]
        assert index_def["mappings"]["fields"]["embedding"]["dimensions"] == 384
        assert index_def["mappings"]["fields"]["embedding"]["similarity"] == "cosine"


class TestMongoDBVectorStoreIntegration:
    """
    Integration tests for MongoDB Vector Store.
    
    These tests require a real MongoDB connection and are skipped if not available.
    To run these tests:
    1. Set MONGODB_URI in your .env
    2. Create a vector search index named "vector_index"
    3. Run: pytest tests/test_mongodb.py -v -m integration
    """
    
    @pytest.fixture
    def mongodb_uri(self):
        """Get MongoDB URI from environment."""
        import os
        uri = os.getenv("MONGODB_URI", "")
        if not uri or "username:password" in uri:
            pytest.skip("MongoDB URI not configured")
        return uri
    
    @pytest.mark.integration
    @patch('src.vector_store.get_settings')
    def test_real_connection(self, mock_get_settings, mongodb_uri):
        """Test real MongoDB connection."""
        mock_settings = Mock()
        mock_settings.vector_store = Mock()
        mock_settings.vector_store.mongodb_uri = mongodb_uri
        mock_settings.vector_store.mongodb_database = "test_rag"
        mock_settings.vector_store.mongodb_collection = "test_chunks"
        mock_settings.vector_store.mongodb_vector_index = "vector_index"
        mock_get_settings.return_value = mock_settings
        
        store = MongoDBVectorStore(
            dimension=384,
            uri=mongodb_uri,
            database="test_rag",
            collection="test_chunks"
        )
        
        # This should not raise
        store._connect()
        
        # Test count
        count = store.count()
        assert isinstance(count, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
