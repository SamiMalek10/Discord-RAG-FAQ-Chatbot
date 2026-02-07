"""
RAG Chain Module

Orchestrates the Retrieval-Augmented Generation pipeline:
1. Query preprocessing
2. Vector search (retrieval)
3. Context building
4. LLM generation
5. Response formatting with sources

This is the core "intelligence" component that ties everything together.

Design Rationale:
- Separates retrieval and generation concerns
- Manages context length for LLM limits
- Provides source attribution (no hallucination)
- Supports conversation memory for follow-ups

RAG Pipeline Flow:
    User Query → Query Embedding → Vector Search → Retrieve Top-K Chunks
    → Build Prompt [Context + Question] → LLM Generation → Return Answer with Sources
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from config.settings import get_settings
from src.embeddings import EmbeddingService
from src.vector_store import VectorStore, SearchResult
from src.llm_service import LLMService, LLMResponse
from src.memory import ConversationMemory

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """
    Complete response from the RAG chain.
    
    Attributes:
        answer: The generated answer text
        sources: List of source documents used
        retrieved_chunks: The actual chunks retrieved
        confidence: Confidence score (based on retrieval similarity)
        query: The original query
        metadata: Additional info (latency, tokens, etc.)
    """
    answer: str
    sources: List[Dict[str, Any]]
    retrieved_chunks: List[str]
    confidence: float
    query: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for API response)."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "retrieved_chunks": self.retrieved_chunks,
            "confidence": self.confidence,
            "query": self.query,
            "metadata": self.metadata,
        }
    
    def format_with_sources(self) -> str:
        """Format answer with source citations."""
        if not self.sources:
            return self.answer
        
        source_text = "\n\n📚 **Sources:**\n"
        unique_sources = {s["source"] for s in self.sources}
        for source in unique_sources:
            source_text += f"- {source}\n"
        
        return self.answer + source_text


class RAGChain:
    """
    Main RAG Chain that orchestrates retrieval and generation.
    
    This is the core component that:
    1. Takes a user query
    2. Retrieves relevant context from the vector store
    3. Builds a prompt with the context
    4. Generates a response using the LLM
    5. Returns the response with source attribution
    
    Example:
        # Initialize components
        embedding_service = EmbeddingService()
        vector_store = VectorStore(embedding_service=embedding_service)
        llm_service = LLMService()
        
        # Create RAG chain
        rag = RAGChain(
            vector_store=vector_store,
            llm_service=llm_service,
        )
        
        # Query
        response = rag.query("What is the bootcamp schedule?")
        print(response.answer)
        print(response.sources)
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm_service: LLMService,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        max_context_length: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the RAG Chain.
        
        Args:
            vector_store: Vector store for retrieval
            llm_service: LLM service for generation
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity score
            max_context_length: Max chars for context
            system_prompt: Custom system prompt
        """
        settings = get_settings()
        
        self.vector_store = vector_store
        self.llm_service = llm_service
        
        # Retrieval settings
        self.top_k = top_k or settings.retrieval.top_k
        self.similarity_threshold = similarity_threshold or settings.retrieval.similarity_threshold
        self.max_context_length = max_context_length or settings.retrieval.max_context_length
        
        # Custom system prompt
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        logger.info(
            f"RAGChain initialized: top_k={self.top_k}, "
            f"threshold={self.similarity_threshold}"
        )
    
    def _default_system_prompt(self) -> str:
        """Get the default system prompt for the AI Bootcamp FAQ bot."""
        return """You are a helpful AI assistant for the PM Accelerator AI Bootcamp program.
Your role is to answer questions about the bootcamp using ONLY the provided context.

CRITICAL INSTRUCTIONS:
1. READ THE CONTEXT CAREFULLY - Extract specific information, dates, names, and details
2. ALWAYS provide an answer if ANY relevant information exists in the context
3. Be SPECIFIC - Include exact weeks, dates, tier names, percentages, and other details from the context
4. STRUCTURE your answers clearly with bullet points or numbered lists when appropriate
5. CITE sources - Mention which document contains the information
6. For schedule questions: Extract exact week numbers and activities
7. For award/tier questions: List ALL tiers with their specific criteria
8. For process questions: Provide step-by-step information

ONLY say "I don't have information" if the context is COMPLETELY unrelated to the question.
Remember: Users need accurate, detailed answers about the AI Bootcamp program."""
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess the user query.
        
        Simple preprocessing to improve retrieval:
        - Strip whitespace
        - Basic normalization
        
        Args:
            query: Raw user query
            
        Returns:
            Preprocessed query
        """
        # Strip whitespace
        query = query.strip()
        
        # Could add more preprocessing here:
        # - Spell correction
        # - Query expansion
        # - Synonym handling
        
        return query
    
    def _build_context(self, results: List[SearchResult]) -> str:
        """
        Build context string from search results with deduplication.
        
        Combines retrieved chunks into a context string,
        removing duplicates and respecting max context length.
        
        Args:
            results: Search results from vector store
            
        Returns:
            Combined context string (deduplicated)
        """
        if not results:
            return ""
        
        context_parts = []
        current_length = 0
        seen_texts = set()  # Track seen content to avoid duplicates
        
        for result in results:
            chunk_text = result.chunk.text.strip()
            source = result.chunk.source
            
            # Skip duplicate content (exact or near-duplicate)
            text_hash = chunk_text[:200]  # Use first 200 chars as fingerprint
            if text_hash in seen_texts:
                logger.debug(f"Skipping duplicate chunk from {source}")
                continue
            seen_texts.add(text_hash)
            
            # Format chunk with source and relevance score
            formatted = f"[Source: {source} | Relevance: {result.score:.1%}]\n{chunk_text}"
            
            # Check if we'd exceed max length
            if current_length + len(formatted) > self.max_context_length:
                # Truncate if needed
                remaining = self.max_context_length - current_length
                if remaining > 100:  # Only add if meaningful content
                    formatted = formatted[:remaining] + "..."
                    context_parts.append(formatted)
                break
            
            context_parts.append(formatted)
            current_length += len(formatted) + 2  # +2 for separator
        
        logger.info(f"Built context with {len(context_parts)} unique chunks (filtered from {len(results)})")
        return "\n\n---\n\n".join(context_parts)
    
    def _calculate_confidence(self, results: List[SearchResult]) -> float:
        """
        Calculate confidence score based on retrieval results.
        
        Uses average similarity of top results as confidence proxy.
        
        Args:
            results: Search results
            
        Returns:
            Confidence score (0-1)
        """
        if not results:
            return 0.0
        
        # Average similarity of retrieved chunks
        avg_similarity = sum(r.score for r in results) / len(results)
        
        # Boost confidence if top result is very relevant
        top_score = results[0].score if results else 0
        
        # Weighted combination
        confidence = 0.6 * avg_similarity + 0.4 * top_score
        
        return min(confidence, 1.0)
    
    def _extract_sources(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """
        Extract source information from results.
        
        Args:
            results: Search results
            
        Returns:
            List of source dictionaries
        """
        sources = []
        seen = set()
        
        for result in results:
            chunk = result.chunk
            source_key = f"{chunk.source}:{chunk.chunk_index}"
            
            if source_key not in seen:
                seen.add(source_key)
                sources.append({
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "score": result.score,
                    "preview": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
                })
        
        return sources
    
    def query(
        self,
        query: str,
        memory: Optional[ConversationMemory] = None,
        include_history: bool = True,
    ) -> RAGResponse:
        """
        Process a query through the RAG pipeline.
        
        This is the main entry point for querying the RAG system.
        
        Args:
            query: User's question
            memory: Optional conversation memory for context
            include_history: Whether to include chat history in prompt
            
        Returns:
            RAGResponse with answer, sources, and metadata
        """
        import time
        start_time = time.time()
        
        # Step 1: Preprocess query
        processed_query = self._preprocess_query(query)
        logger.debug(f"Processing query: {processed_query}")
        
        # Step 2: Retrieve relevant chunks
        results = self.vector_store.search(
            query=processed_query,
            top_k=self.top_k,
            threshold=self.similarity_threshold,
        )
        
        retrieval_time = time.time() - start_time
        logger.debug(f"Retrieved {len(results)} chunks in {retrieval_time:.2f}s")
        
        # Step 3: Build context from results
        context = self._build_context(results)
        
        # Step 4: Handle case with no relevant results
        if not results or not context:
            return RAGResponse(
                answer="I couldn't find relevant information in the knowledge base to answer your question. Could you try rephrasing or ask about something related to the AI Bootcamp program?",
                sources=[],
                retrieved_chunks=[],
                confidence=0.0,
                query=query,
                metadata={
                    "retrieval_time": retrieval_time,
                    "total_time": time.time() - start_time,
                    "chunks_found": 0,
                },
            )
        
        # Step 5: Build system prompt with conversation history
        system_prompt = self.system_prompt
        if memory and include_history and len(memory) > 0:
            history = memory.get_context_string()
            system_prompt += f"\n\nPrevious conversation:\n{history}"
        
        # Step 6: Generate response using LLM
        generation_start = time.time()
        
        llm_response = self.llm_service.generate_with_context(
            query=processed_query,
            context=context,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower for factual responses
        )
        
        generation_time = time.time() - generation_start
        
        # Step 7: Build response
        total_time = time.time() - start_time
        
        response = RAGResponse(
            answer=llm_response.content,
            sources=self._extract_sources(results),
            retrieved_chunks=[r.chunk.text for r in results],
            confidence=self._calculate_confidence(results),
            query=query,
            metadata={
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "chunks_found": len(results),
                "model": llm_response.model,
                "usage": llm_response.usage,
            },
        )
        
        # Step 8: Update memory if provided
        if memory:
            memory.add_user_message(query)
            memory.add_assistant_message(
                llm_response.content,
                metadata={"sources": [s["source"] for s in response.sources]},
            )
        
        logger.info(
            f"RAG query completed in {total_time:.2f}s "
            f"(retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)"
        )
        
        return response
    
    def query_simple(self, query: str) -> str:
        """
        Simple query that returns just the answer string.
        
        Convenience method for simple use cases.
        
        Args:
            query: User's question
            
        Returns:
            Answer string
        """
        response = self.query(query)
        return response.answer
    
    def get_relevant_chunks(
        self, 
        query: str, 
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Retrieve relevant chunks without generating a response.
        
        Useful for debugging or showing retrieved context.
        
        Args:
            query: User's question
            top_k: Number of results
            
        Returns:
            List of SearchResult objects
        """
        processed_query = self._preprocess_query(query)
        return self.vector_store.search(
            query=processed_query,
            top_k=top_k or self.top_k,
            threshold=self.similarity_threshold,
        )


def create_rag_chain(
    embedding_provider: str = "local",
    llm_provider: str = "ollama",
    vector_store_provider: str = "faiss",
) -> RAGChain:
    """
    Factory function to create a fully configured RAG chain.
    
    Convenience function that creates all necessary components.
    
    Args:
        embedding_provider: "local" or "openai"
        llm_provider: "ollama", "openai", or "gemini"
        vector_store_provider: "faiss" or "mongodb"
        
    Returns:
        Configured RAGChain instance
    """
    # Create components
    embedding_service = EmbeddingService(provider=embedding_provider)
    vector_store = VectorStore(
        embedding_service=embedding_service,
        provider=vector_store_provider,
    )
    llm_service = LLMService(provider=llm_provider)
    
    # Create and return chain
    return RAGChain(
        vector_store=vector_store,
        llm_service=llm_service,
    )
