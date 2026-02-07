"""
Discord RAG FAQ Chatbot - Core Source Module

This module contains the core RAG components:
- DocumentChunker: Document segmentation with metadata preservation
- EmbeddingService: Abstract embedding generation (local vs cloud)
- VectorStore: Vector database interface (FAISS/MongoDB)
- LLMService: LLM provider abstraction (Ollama/OpenAI/Gemini)
- ConversationMemory: Multi-turn conversation management
- RAGChain: Orchestrates retrieval + generation
- RAGAgent: API interface for Backend Engineers
"""

from .chunker import DocumentChunker, Chunk
from .embeddings import EmbeddingService
from .vector_store import VectorStore, SearchResult
from .llm_service import LLMService, LLMResponse
from .memory import ConversationMemory, ConversationManager, Message
from .rag_chain import RAGChain, RAGResponse, create_rag_chain
from .rag_agent import RAGAgent, create_agent

__all__ = [
    # Phase 1: Core Components
    "DocumentChunker",
    "Chunk",
    "EmbeddingService",
    "VectorStore",
    "SearchResult",
    # Phase 2: RAG Pipeline
    "LLMService",
    "LLMResponse",
    "ConversationMemory",
    "ConversationManager",
    "Message",
    "RAGChain",
    "RAGResponse",
    "RAGAgent",
    # Factory functions
    "create_rag_chain",
    "create_agent",
]
