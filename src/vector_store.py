"""
Vector Store Module

Provides vector database functionality for storing and searching embeddings.
Supports two backends:
- FAISS: Local, fast, for development/prototyping
- MongoDB Atlas: Production, scalable, with vector search

Design Rationale:
- Abstract interface for easy backend switching
- FAISS for quick local development (no external dependencies)
- MongoDB Atlas for production (aligns with workshop, scalable)
- Unified search interface regardless of backend

Schema (stored per chunk):
- text: Original text content
- embedding: Vector representation
- source: Document filename
- chunk_id: Unique identifier
- metadata: Additional info (page, section, etc.)
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from config.settings import get_settings, VectorStoreConfig
from src.chunker import Chunk
from src.embeddings import EmbeddingService, cosine_similarity

# Configure logging
logger = logging.getLogger(__name__)


class SearchResult:
    """
    Represents a single search result.
    
    Attributes:
        chunk: The retrieved Chunk object
        score: Similarity score (0-1, higher is better)
        rank: Position in results (1-indexed)
    """
    
    def __init__(self, chunk: Chunk, score: float, rank: int = 0):
        self.chunk = chunk
        self.score = score
        self.rank = rank
    
    def __repr__(self) -> str:
        return (
            f"SearchResult(source='{self.chunk.source}', "
            f"score={self.score:.4f}, rank={self.rank})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "rank": self.rank,
        }


class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.
    
    All implementations must provide:
    - add_chunks: Add chunks with embeddings
    - search: Find similar chunks
    - delete: Remove chunks
    - clear: Remove all data
    """
    
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> int:
        """
        Add chunks with embeddings to the store.
        
        Args:
            chunks: List of Chunk objects (must have embeddings)
            
        Returns:
            Number of chunks successfully added
        """
        pass
    
    @abstractmethod
    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[SearchResult]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Minimum similarity score
            
        Returns:
            List of SearchResult objects, sorted by score descending
        """
        pass
    
    @abstractmethod
    def delete(self, chunk_ids: List[str]) -> int:
        """
        Delete chunks by their IDs.
        
        Args:
            chunk_ids: List of chunk IDs to delete
            
        Returns:
            Number of chunks deleted
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Remove all data from the store."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Return total number of chunks in store."""
        pass


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based vector store for local development.
    
    FAISS (Facebook AI Similarity Search) provides:
    - Fast similarity search
    - In-memory operation
    - Optional persistence to disk
    - No external database needed
    
    Best for:
    - Development and prototyping
    - Small to medium datasets
    - Offline operation
    """
    
    def __init__(
        self,
        dimension: int,
        index_path: Optional[str] = None,
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Embedding dimension (must match your model)
            index_path: Path to save/load index (optional)
        """
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        
        # Storage for chunks (FAISS only stores vectors)
        self._chunks: Dict[int, Chunk] = {}
        self._id_to_index: Dict[str, int] = {}  # chunk_id -> faiss index
        self._next_index = 0
        
        # Initialize FAISS index
        self._index = None
        self._init_index()
        
        # Try to load existing index
        if self.index_path and self.index_path.exists():
            self._load()
        
        logger.info(
            f"FAISSVectorStore initialized: dimension={dimension}, "
            f"index_path={index_path}"
        )
    
    def _init_index(self):
        """Initialize the FAISS index."""
        try:
            import faiss
            
            # Using IndexFlatIP (Inner Product) for cosine similarity
            # Vectors should be normalized for this to work as cosine similarity
            self._index = faiss.IndexFlatIP(self.dimension)
            
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for FAISS vector store. "
                "Install with: pip install faiss-cpu"
            )
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms
    
    def add_chunks(self, chunks: List[Chunk]) -> int:
        """
        Add chunks to the FAISS index.
        
        Args:
            chunks: List of Chunk objects with embeddings
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        # Filter chunks that have embeddings
        valid_chunks = [c for c in chunks if c.embedding is not None]
        
        if not valid_chunks:
            logger.warning("No chunks with embeddings to add")
            return 0
        
        # Prepare vectors
        vectors = np.array(
            [c.embedding for c in valid_chunks], 
            dtype=np.float32
        )
        
        # Normalize for cosine similarity
        vectors = self._normalize(vectors)
        
        # Add to FAISS index
        self._index.add(vectors)
        
        # Store chunk data
        for chunk in valid_chunks:
            self._chunks[self._next_index] = chunk
            self._id_to_index[chunk.chunk_id] = self._next_index
            self._next_index += 1
        
        logger.info(f"Added {len(valid_chunks)} chunks to FAISS index")
        
        # Auto-save if path configured
        if self.index_path:
            self._save()
        
        return len(valid_chunks)
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[SearchResult]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of SearchResult objects
        """
        if self._index.ntotal == 0:
            logger.warning("Search on empty index")
            return []
        
        # Prepare query vector
        query_vector = np.array([query_embedding], dtype=np.float32)
        query_vector = self._normalize(query_vector)
        
        # Search
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vector, k)
        
        # Build results
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx < 0:  # FAISS returns -1 for not found
                continue
            
            # Convert inner product to similarity score (already normalized)
            similarity = float(score)
            
            if similarity < threshold:
                continue
            
            chunk = self._chunks.get(idx)
            if chunk:
                results.append(SearchResult(
                    chunk=chunk,
                    score=similarity,
                    rank=rank,
                ))
        
        logger.debug(f"Search returned {len(results)} results")
        return results
    
    def delete(self, chunk_ids: List[str]) -> int:
        """
        Delete chunks by ID.
        
        Note: FAISS doesn't support efficient deletion.
        We mark as deleted and rebuild periodically.
        
        Args:
            chunk_ids: Chunk IDs to delete
            
        Returns:
            Number deleted
        """
        deleted = 0
        for chunk_id in chunk_ids:
            if chunk_id in self._id_to_index:
                idx = self._id_to_index[chunk_id]
                if idx in self._chunks:
                    del self._chunks[idx]
                    deleted += 1
                del self._id_to_index[chunk_id]
        
        # Rebuild index if significant deletions
        if deleted > 0 and deleted > len(self._chunks) * 0.1:
            self._rebuild_index()
        
        return deleted
    
    def _rebuild_index(self):
        """Rebuild the index (needed after deletions)."""
        import faiss
        
        logger.info("Rebuilding FAISS index...")
        
        # Get remaining chunks
        remaining = list(self._chunks.values())
        
        # Reset
        self._index = faiss.IndexFlatIP(self.dimension)
        self._chunks = {}
        self._id_to_index = {}
        self._next_index = 0
        
        # Re-add
        if remaining:
            self.add_chunks(remaining)
    
    def clear(self) -> None:
        """Clear all data from the store."""
        import faiss
        
        self._index = faiss.IndexFlatIP(self.dimension)
        self._chunks = {}
        self._id_to_index = {}
        self._next_index = 0
        
        if self.index_path:
            self._save()
        
        logger.info("FAISS index cleared")
    
    def count(self) -> int:
        """Return number of vectors in index."""
        return len(self._chunks)
    
    def _save(self):
        """Save index and metadata to disk."""
        import faiss
        
        if not self.index_path:
            return
        
        # Create directory if needed
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self._index, str(self.index_path))
        
        # Save chunk metadata
        metadata_path = self.index_path.with_suffix(".json")
        metadata = {
            "chunks": {str(k): v.to_dict() for k, v in self._chunks.items()},
            "id_to_index": self._id_to_index,
            "next_index": self._next_index,
            "dimension": self.dimension,
        }
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"Saved FAISS index to {self.index_path}")
    
    def _load(self):
        """Load index and metadata from disk."""
        import faiss
        
        if not self.index_path or not self.index_path.exists():
            return
        
        try:
            # Load FAISS index
            self._index = faiss.read_index(str(self.index_path))
            
            # Load metadata
            metadata_path = self.index_path.with_suffix(".json")
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                
                self._chunks = {
                    int(k): Chunk.from_dict(v) 
                    for k, v in metadata["chunks"].items()
                }
                self._id_to_index = metadata["id_to_index"]
                self._next_index = metadata["next_index"]
            
            logger.info(
                f"Loaded FAISS index with {self._index.ntotal} vectors"
            )
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            self._init_index()


class MongoDBVectorStore(BaseVectorStore):
    """
    MongoDB Atlas Vector Store for production use.
    
    MongoDB Atlas Vector Search provides:
    - Scalable cloud-hosted vector search
    - Integrated with MongoDB document storage
    - Supports hybrid queries (vector + filter)
    - Managed infrastructure
    
    Requires:
    - MongoDB Atlas cluster with Vector Search enabled
    - Vector search index created on the collection
    
    Best for:
    - Production deployments
    - Large datasets
    - When you need filtering capabilities
    """
    
    def __init__(
        self,
        dimension: int,
        uri: Optional[str] = None,
        database: Optional[str] = None,
        collection: Optional[str] = None,
        vector_index: Optional[str] = None,
    ):
        """
        Initialize MongoDB Vector Store.
        
        Args:
            dimension: Embedding dimension
            uri: MongoDB connection URI (or from env)
            database: Database name
            collection: Collection name
            vector_index: Name of the vector search index
        """
        settings = get_settings()
        config = settings.vector_store
        
        self.dimension = dimension
        self.uri = uri or config.mongodb_uri
        self.database_name = database or config.mongodb_database
        self.collection_name = collection or config.mongodb_collection
        self.vector_index = vector_index or config.mongodb_vector_index
        
        self._client = None
        self._db = None
        self._collection = None
        
        logger.info(
            f"MongoDBVectorStore initialized: db={self.database_name}, "
            f"collection={self.collection_name}"
        )
    
    def _connect(self):
        """Establish connection to MongoDB."""
        if self._collection is not None:
            return
        
        if not self.uri:
            raise ValueError(
                "MongoDB URI not configured. Set MONGODB_URI environment variable."
            )
        
        try:
            from pymongo import MongoClient
            
            self._client = MongoClient(self.uri)
            self._db = self._client[self.database_name]
            self._collection = self._db[self.collection_name]
            
            # Test connection
            self._client.admin.command("ping")
            
            logger.info("Connected to MongoDB Atlas")
            
        except ImportError:
            raise ImportError(
                "pymongo is required for MongoDB. "
                "Install with: pip install 'pymongo[srv]'"
            )
    
    def add_chunks(self, chunks: List[Chunk]) -> int:
        """
        Add chunks to MongoDB.
        
        Args:
            chunks: List of Chunk objects with embeddings
            
        Returns:
            Number of chunks added
        """
        self._connect()
        
        if not chunks:
            return 0
        
        # Filter chunks with embeddings
        valid_chunks = [c for c in chunks if c.embedding is not None]
        
        if not valid_chunks:
            logger.warning("No chunks with embeddings to add")
            return 0
        
        # Prepare documents
        documents = []
        for chunk in valid_chunks:
            doc = {
                "_id": chunk.chunk_id,
                "text": chunk.text,
                "embedding": chunk.embedding,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "metadata": chunk.metadata,
            }
            documents.append(doc)
        
        # Upsert documents (update if exists, insert if new)
        from pymongo import UpdateOne
        
        operations = [
            UpdateOne(
                {"_id": doc["_id"]},
                {"$set": doc},
                upsert=True
            )
            for doc in documents
        ]
        
        result = self._collection.bulk_write(operations)
        added = result.upserted_count + result.modified_count
        
        logger.info(f"Added/updated {added} chunks in MongoDB")
        return added
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar chunks using Atlas Vector Search.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            threshold: Minimum similarity score
            filter_dict: Optional MongoDB filter
            
        Returns:
            List of SearchResult objects
        """
        self._connect()
        
        # Build the vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_index,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 10,  # Over-fetch for filtering
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "text": 1,
                    "source": 1,
                    "chunk_index": 1,
                    "total_chunks": 1,
                    "metadata": 1,
                    "embedding": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            }
        ]
        
        # Add filter if provided
        if filter_dict:
            pipeline.insert(1, {"$match": filter_dict})
        
        # Execute search
        results = list(self._collection.aggregate(pipeline))
        
        # Build SearchResult objects
        search_results = []
        for rank, doc in enumerate(results, 1):
            if doc.get("score", 0) < threshold:
                continue
            
            chunk = Chunk(
                text=doc["text"],
                chunk_id=doc["_id"],
                source=doc["source"],
                chunk_index=doc["chunk_index"],
                total_chunks=doc.get("total_chunks", 0),
                metadata=doc.get("metadata", {}),
                embedding=doc.get("embedding"),
            )
            
            search_results.append(SearchResult(
                chunk=chunk,
                score=doc.get("score", 0),
                rank=rank,
            ))
        
        logger.debug(f"MongoDB search returned {len(search_results)} results")
        return search_results
    
    def delete(self, chunk_ids: List[str]) -> int:
        """Delete chunks by ID."""
        self._connect()
        
        result = self._collection.delete_many({"_id": {"$in": chunk_ids}})
        logger.info(f"Deleted {result.deleted_count} chunks from MongoDB")
        return result.deleted_count
    
    def clear(self) -> None:
        """Clear all documents from collection."""
        self._connect()
        
        result = self._collection.delete_many({})
        logger.info(f"Cleared {result.deleted_count} documents from MongoDB")
    
    def count(self) -> int:
        """Return document count."""
        self._connect()
        return self._collection.count_documents({})
    
    def create_vector_index(self):
        """
        Create the vector search index.
        
        Note: This usually needs to be done via Atlas UI or CLI.
        This method provides guidance.
        """
        index_definition = {
            "mappings": {
                "dynamic": True,
                "fields": {
                    "embedding": {
                        "type": "knnVector",
                        "dimensions": self.dimension,
                        "similarity": "cosine",
                    }
                }
            }
        }
        
        logger.info(
            f"To create the vector index '{self.vector_index}', "
            f"use the following definition in Atlas:\n"
            f"{json.dumps(index_definition, indent=2)}"
        )
        return index_definition


class VectorStore:
    """
    Main Vector Store class with unified interface.
    
    This is the class that other components should use.
    It handles provider selection based on configuration.
    
    Example:
        # With embedding service
        store = VectorStore(embedding_service=embedding_service)
        
        # Add chunks (will auto-embed if needed)
        store.add_chunks(chunks)
        
        # Search
        results = store.search("What is the bootcamp schedule?")
    """
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        provider: Optional[str] = None,
        config: Optional[VectorStoreConfig] = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            embedding_service: EmbeddingService for embedding generation
            provider: "faiss" or "mongodb" (default from config)
            config: Optional VectorStoreConfig
        """
        settings = get_settings()
        self.config = config or settings.vector_store
        
        # Embedding service (for auto-embedding)
        self.embedding_service = embedding_service or EmbeddingService()
        dimension = self.embedding_service.dimension
        
        # Determine provider
        provider = provider or self.config.provider
        
        # Initialize backend
        if provider == "faiss":
            self._store = FAISSVectorStore(
                dimension=dimension,
                index_path=self.config.faiss_index_path,
            )
        elif provider == "mongodb":
            self._store = MongoDBVectorStore(
                dimension=dimension,
                uri=self.config.mongodb_uri,
                database=self.config.mongodb_database,
                collection=self.config.mongodb_collection,
                vector_index=self.config.mongodb_vector_index,
            )
        else:
            raise ValueError(f"Unknown vector store provider: {provider}")
        
        logger.info(f"VectorStore initialized with {provider} backend")
    
    def add_chunks(
        self, 
        chunks: List[Chunk], 
        embed: bool = True
    ) -> int:
        """
        Add chunks to the store.
        
        Args:
            chunks: List of Chunk objects
            embed: Whether to generate embeddings (default True)
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        # Generate embeddings if needed
        if embed:
            chunks_to_embed = [c for c in chunks if c.embedding is None]
            
            if chunks_to_embed:
                logger.info(f"Generating embeddings for {len(chunks_to_embed)} chunks")
                texts = [c.text for c in chunks_to_embed]
                embeddings = self.embedding_service.embed_batch(texts)
                
                for chunk, embedding in zip(chunks_to_embed, embeddings):
                    chunk.embedding = embedding
        
        return self._store.add_chunks(chunks)
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search for chunks relevant to a query.
        
        Args:
            query: User's question/search query
            top_k: Number of results (default from config)
            threshold: Minimum similarity (default from config)
            
        Returns:
            List of SearchResult objects
        """
        settings = get_settings()
        top_k = top_k or settings.retrieval.top_k
        threshold = threshold or settings.retrieval.similarity_threshold
        
        # Embed the query
        query_embedding = self.embedding_service.embed_query(query)
        
        # Search
        return self._store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold,
        )
    
    def delete(self, chunk_ids: List[str]) -> int:
        """Delete chunks by ID."""
        return self._store.delete(chunk_ids)
    
    def clear(self) -> None:
        """Clear all data."""
        self._store.clear()
    
    def count(self) -> int:
        """Return chunk count."""
        return self._store.count()
    
    @property
    def provider(self) -> str:
        """Return the backend provider name."""
        return self.config.provider
