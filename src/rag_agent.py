"""
RAG Agent Module

Provides the API interface for Backend Engineers to integrate the RAG system.
This is the main entry point that encapsulates all RAG functionality.

API Contract (as specified in assignment):
    class RAGAgent:
        def query(self, question: str, chat_history: list = None) -> dict
        def ingest_document(self, file_path: str, metadata: dict) -> bool

Design Rationale:
- Clean, simple interface for Backend team
- Handles all complexity internally
- Thread-safe for concurrent Discord usage
- Proper error handling with graceful fallbacks
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import threading

from config.settings import get_settings
from src.chunker import DocumentChunker, Chunk
from src.embeddings import EmbeddingService
from src.vector_store import VectorStore
from src.llm_service import LLMService
from src.rag_chain import RAGChain, RAGResponse
from src.memory import ConversationMemory, ConversationManager

logger = logging.getLogger(__name__)


class RAGAgent:
    """
    Main RAG Agent - the public API for the Discord RAG FAQ Chatbot.
    
    This class provides a clean interface for Backend Engineers to:
    1. Query the knowledge base
    2. Ingest new documents
    3. Manage conversations
    
    The agent handles all internal complexity including:
    - Document chunking
    - Embedding generation
    - Vector storage and search
    - LLM generation
    - Conversation memory
    
    Example:
        # Initialize the agent
        agent = RAGAgent()
        
        # Ingest documents
        agent.ingest_document("path/to/faq.pdf")
        agent.ingest_directory("data/knowledge_base")
        
        # Query
        result = agent.query("What is the bootcamp schedule?")
        print(result["answer"])
        print(result["sources"])
        
        # Query with conversation history
        result = agent.query(
            "What about week 4?",
            conversation_id="user_123"
        )
    
    API Contract:
        query() returns:
        {
            "answer": str,
            "sources": list[dict],
            "confidence": float,
            "retrieved_chunks": list[str]
        }
    """
    
    def __init__(
        self,
        embedding_provider: Optional[str] = None,
        llm_provider: Optional[str] = None,
        vector_store_provider: Optional[str] = None,
        auto_load_knowledge_base: bool = False,
    ):
        """
        Initialize the RAG Agent.
        
        Args:
            embedding_provider: "local" or "openai" (default from config)
            llm_provider: "ollama", "openai", or "gemini" (default from config)
            vector_store_provider: "faiss" or "mongodb" (default from config)
            auto_load_knowledge_base: Load documents from data/knowledge_base on init
        """
        settings = get_settings()
        
        # Initialize components
        logger.info("Initializing RAG Agent...")
        
        # Embedding service
        self._embedding_service = EmbeddingService(
            provider=embedding_provider or settings.embedding.provider
        )
        
        # Vector store
        self._vector_store = VectorStore(
            embedding_service=self._embedding_service,
            provider=vector_store_provider or settings.vector_store.provider,
        )
        
        # LLM service
        self._llm_service = LLMService(
            provider=llm_provider or settings.llm.provider
        )
        
        # Document chunker
        self._chunker = DocumentChunker()
        
        # RAG chain
        self._rag_chain = RAGChain(
            vector_store=self._vector_store,
            llm_service=self._llm_service,
        )
        
        # Conversation manager (for multi-user support)
        self._conversation_manager = ConversationManager()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Auto-load knowledge base if requested
        if auto_load_knowledge_base:
            self._auto_load_knowledge_base()
        
        logger.info(
            f"RAG Agent initialized: "
            f"embedding={self._embedding_service.provider_name}, "
            f"llm={self._llm_service.provider_name}, "
            f"vector_store={self._vector_store.provider}"
        )
    
    def _auto_load_knowledge_base(self) -> None:
        """Load documents from the default knowledge base directory."""
        settings = get_settings()
        kb_path = settings.knowledge_base_dir
        
        if kb_path.exists() and kb_path.is_dir():
            # Check if there are documents
            docs = list(kb_path.glob("*.*"))
            supported = [d for d in docs if d.suffix.lower() in [".pdf", ".txt", ".docx", ".md"]]
            
            if supported:
                logger.info(f"Auto-loading {len(supported)} documents from knowledge base")
                self.ingest_directory(str(kb_path))
            else:
                logger.warning(f"No supported documents found in {kb_path}")
        else:
            logger.warning(f"Knowledge base directory not found: {kb_path}")
    
    def query(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        This is the main API method for querying the knowledge base.
        
        Args:
            question: User's question
            chat_history: Optional list of previous messages
                         Format: [{"role": "user", "content": "..."}, ...]
            conversation_id: Optional ID for conversation continuity
            
        Returns:
            Dictionary with:
            - answer: str - The generated answer
            - sources: list[dict] - Source documents used
            - confidence: float - Confidence score (0-1)
            - retrieved_chunks: list[str] - Retrieved text chunks
            
        Example:
            result = agent.query("What is the bootcamp schedule?")
            print(result["answer"])
        """
        # Get or create conversation memory
        memory = None
        if conversation_id:
            memory = self._conversation_manager.get_memory(conversation_id)
            
            # Load chat history if provided
            if chat_history and len(memory) == 0:
                for msg in chat_history:
                    if msg["role"] == "user":
                        memory.add_user_message(msg["content"])
                    elif msg["role"] == "assistant":
                        memory.add_assistant_message(msg["content"])
        
        try:
            # Execute RAG query
            response = self._rag_chain.query(
                query=question,
                memory=memory,
            )
            
            # Return API contract format
            return {
                "answer": response.answer,
                "sources": response.sources,
                "confidence": response.confidence,
                "retrieved_chunks": response.retrieved_chunks,
            }
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {
                "answer": "I'm sorry, I encountered an error processing your question. Please try again.",
                "sources": [],
                "confidence": 0.0,
                "retrieved_chunks": [],
                "error": str(e),
            }
    
    def ingest_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Ingest a document into the knowledge base.
        
        Processes the document through:
        1. Loading and parsing
        2. Chunking with metadata
        3. Embedding generation
        4. Vector store insertion
        
        Args:
            file_path: Path to the document file
            metadata: Optional additional metadata
            
        Returns:
            True if successful, False otherwise
            
        Example:
            success = agent.ingest_document("faq.pdf", {"category": "FAQ"})
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                return False
            
            logger.info(f"Ingesting document: {path.name}")
            
            # Chunk the document
            chunks = self._chunker.process_document(
                file_path=path,
                additional_metadata=metadata,
            )
            
            if not chunks:
                logger.warning(f"No chunks created from {path.name}")
                return False
            
            # Add to vector store (handles embedding)
            with self._lock:
                added = self._vector_store.add_chunks(chunks)
            
            logger.info(f"Ingested {added} chunks from {path.name}")
            return added > 0
            
        except Exception as e:
            logger.error(f"Error ingesting {file_path}: {e}")
            return False
    
    def ingest_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest all documents from a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search subdirectories
            metadata: Optional metadata for all documents
            
        Returns:
            Dictionary with ingestion statistics
            
        Example:
            stats = agent.ingest_directory("data/knowledge_base")
            print(f"Ingested {stats['documents_processed']} documents")
        """
        path = Path(directory_path)
        
        if not path.exists() or not path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            return {"success": False, "error": "Directory not found"}
        
        logger.info(f"Ingesting directory: {path}")
        
        # Find all supported files
        supported_extensions = [".pdf", ".txt", ".docx", ".md"]
        pattern = "**/*" if recursive else "*"
        
        files = []
        for ext in supported_extensions:
            files.extend(path.glob(f"{pattern}{ext}"))
        
        # Process each file
        results = {
            "success": True,
            "documents_processed": 0,
            "documents_failed": 0,
            "total_chunks": 0,
            "files": [],
        }
        
        for file_path in files:
            try:
                chunks = self._chunker.process_document(
                    file_path=file_path,
                    additional_metadata=metadata,
                )
                
                if chunks:
                    with self._lock:
                        added = self._vector_store.add_chunks(chunks)
                    
                    results["documents_processed"] += 1
                    results["total_chunks"] += added
                    results["files"].append({
                        "file": file_path.name,
                        "chunks": added,
                        "status": "success",
                    })
                else:
                    results["documents_failed"] += 1
                    results["files"].append({
                        "file": file_path.name,
                        "chunks": 0,
                        "status": "no_chunks",
                    })
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results["documents_failed"] += 1
                results["files"].append({
                    "file": file_path.name,
                    "chunks": 0,
                    "status": "error",
                    "error": str(e),
                })
        
        logger.info(
            f"Directory ingestion complete: "
            f"{results['documents_processed']} docs, "
            f"{results['total_chunks']} chunks"
        )
        
        return results
    
    def ingest_text(
        self,
        text: str,
        source_name: str = "direct_input",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Ingest raw text directly into the knowledge base.
        
        Useful for adding content without a file.
        
        Args:
            text: The text content to ingest
            source_name: Name to use as source
            metadata: Optional metadata
            
        Returns:
            True if successful
        """
        try:
            chunks = self._chunker.process_text(
                text=text,
                source_name=source_name,
                metadata=metadata,
            )
            
            if chunks:
                with self._lock:
                    added = self._vector_store.add_chunks(chunks)
                logger.info(f"Ingested {added} chunks from text")
                return added > 0
            
            return False
            
        except Exception as e:
            logger.error(f"Error ingesting text: {e}")
            return False
    
    def clear_knowledge_base(self) -> bool:
        """
        Clear all documents from the knowledge base.
        
        WARNING: This removes all ingested content!
        
        Returns:
            True if successful
        """
        try:
            with self._lock:
                self._vector_store.clear()
            logger.info("Knowledge base cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system.
        
        Returns:
            Dictionary with system statistics
        """
        return {
            "knowledge_base": {
                "total_chunks": self._vector_store.count(),
                "provider": self._vector_store.provider,
            },
            "embedding": {
                "provider": self._embedding_service.provider_name,
                "model": self._embedding_service.model_name,
                "dimension": self._embedding_service.dimension,
            },
            "llm": {
                "provider": self._llm_service.provider_name,
                "model": self._llm_service.model_name,
            },
            "conversations": {
                "active": len(self._conversation_manager),
            },
        }
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear conversation history for a specific ID.
        
        Args:
            conversation_id: The conversation to clear
            
        Returns:
            True if cleared, False if not found
        """
        return self._conversation_manager.delete_memory(conversation_id)
    
    def search_similar(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar content without generating a response.
        
        Useful for debugging or showing what the system found.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of similar chunks with scores
        """
        results = self._rag_chain.get_relevant_chunks(query, top_k=top_k)
        
        return [
            {
                "text": r.chunk.text,
                "source": r.chunk.source,
                "score": r.score,
                "chunk_id": r.chunk.chunk_id,
            }
            for r in results
        ]


# Convenience function for quick initialization
def create_agent(
    auto_load: bool = True,
    **kwargs,
) -> RAGAgent:
    """
    Create a RAG Agent with sensible defaults.
    
    Args:
        auto_load: Whether to auto-load knowledge base
        **kwargs: Additional arguments for RAGAgent
        
    Returns:
        Configured RAGAgent instance
    """
    return RAGAgent(
        auto_load_knowledge_base=auto_load,
        **kwargs,
    )
