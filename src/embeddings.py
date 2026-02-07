"""
Embedding Service Module

Provides an abstraction layer for embedding generation, supporting both:
- Local: Sentence Transformers (all-MiniLM-L6-v2) - Free, no API key needed
- Cloud: OpenAI (text-embedding-3-small) - Requires API key

Design Rationale:
- Abstract interface allows easy switching between providers
- Local option for development/prototyping (faster, free)
- Cloud option for production (potentially better quality)
- Comparison capability to evaluate model performance

Embedding Dimensions:
- all-MiniLM-L6-v2: 384 dimensions
- all-mpnet-base-v2: 768 dimensions
- text-embedding-3-small: 1536 dimensions
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np

from config.settings import get_settings, EmbeddingConfig
import os
from huggingface_hub import login

if hf_token := os.getenv("HF_TOKEN"):
    login(token=hf_token)

# Configure logging
logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    
    All embedding providers must implement:
    - embed_text: Embed a single text string
    - embed_batch: Embed multiple texts efficiently
    - dimension: Return the embedding dimension
    """
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of embeddings produced."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the embedding model."""
        pass


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """
    Local embedding provider using Sentence Transformers.
    
    Benefits:
    - Free to use (no API costs)
    - Fast inference (runs locally)
    - No internet required
    - Good quality for most use cases
    
    Models:
    - all-MiniLM-L6-v2: Fast, 384 dims (default)
    - all-mpnet-base-v2: Better quality, 768 dims
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the local embedding provider.
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        self._model_name = model_name
        self._model = None
        self._dimension = None
        
        logger.info(f"Initializing LocalEmbeddingProvider with model: {model_name}")
    
    def _load_model(self):
        """Lazy load the model (only when first needed)."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                logger.info(f"Loading sentence-transformers model: {self._model_name}")
                self._model = SentenceTransformer(self._model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded. Embedding dimension: {self._dimension}")
                
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                )
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats (embedding vector)
        """
        self._load_model()
        
        # Sentence transformers returns numpy array
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Uses batching for efficiency.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        self._load_model()
        
        if not texts:
            return []
        
        logger.debug(f"Embedding batch of {len(texts)} texts")
        
        # Batch encode for efficiency
        embeddings = self._model.encode(
            texts, 
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10,
            batch_size=32,
        )
        
        return embeddings.tolist()
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        self._load_model()
        return self._dimension
    
    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model_name


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI embedding provider using the embeddings API.
    
    Benefits:
    - High quality embeddings
    - Large context window
    - Well-tested at scale
    
    Considerations:
    - Requires API key
    - Costs money per token
    - Requires internet connection
    
    Models:
    - text-embedding-3-small: 1536 dims (cheaper)
    - text-embedding-3-large: 3072 dims (better quality)
    - text-embedding-ada-002: 1536 dims (legacy)
    """
    
    # Model dimensions mapping
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self, 
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the OpenAI embedding provider.
        
        Args:
            model_name: Name of the OpenAI embedding model
            api_key: OpenAI API key (or from environment)
        """
        self._model_name = model_name
        self._api_key = api_key
        self._client = None
        
        if model_name not in self.MODEL_DIMENSIONS:
            logger.warning(
                f"Unknown model {model_name}, assuming 1536 dimensions. "
                f"Known models: {list(self.MODEL_DIMENSIONS.keys())}"
            )
        
        logger.info(f"Initializing OpenAIEmbeddingProvider with model: {model_name}")
    
    def _get_client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                
                # Get API key from parameter, env, or settings
                api_key = self._api_key
                if not api_key:
                    import os
                    api_key = os.getenv("OPENAI_API_KEY")
                
                if not api_key:
                    settings = get_settings()
                    api_key = settings.embedding.openai_api_key
                
                if not api_key:
                    raise ValueError(
                        "OpenAI API key not found. Set OPENAI_API_KEY environment "
                        "variable or pass api_key parameter."
                    )
                
                self._client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized")
                
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAI embeddings. "
                    "Install with: pip install openai"
                )
        
        return self._client
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using OpenAI API.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats (embedding vector)
        """
        client = self._get_client()
        
        response = client.embeddings.create(
            input=text,
            model=self._model_name,
        )
        
        return response.data[0].embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using OpenAI API.
        
        OpenAI API supports batching natively.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        client = self._get_client()
        
        logger.debug(f"Embedding batch of {len(texts)} texts via OpenAI")
        
        # OpenAI supports up to 2048 inputs per request
        # For safety, we'll batch in groups of 100
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = client.embeddings.create(
                input=batch,
                model=self._model_name,
            )
            
            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [item.embedding for item in sorted_data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.MODEL_DIMENSIONS.get(self._model_name, 1536)
    
    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model_name


class EmbeddingService:
    """
    Main embedding service that provides a unified interface.
    
    This is the class that other components should use.
    It handles provider selection based on configuration.
    
    Example:
        service = EmbeddingService()  # Uses config
        embedding = service.embed_text("Hello world")
        embeddings = service.embed_batch(["text1", "text2"])
        
        # Or specify provider explicitly
        service = EmbeddingService(provider="openai")
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        config: Optional[EmbeddingConfig] = None,
    ):
        """
        Initialize the embedding service.
        
        Args:
            provider: "local" or "openai" (default from config)
            config: Optional EmbeddingConfig instance
        """
        settings = get_settings()
        self.config = config or settings.embedding
        
        # Determine provider
        provider = provider or self.config.provider
        
        # Initialize the appropriate provider
        if provider == "local":
            self._provider = LocalEmbeddingProvider(
                model_name=self.config.local_model
            )
        elif provider == "openai":
            self._provider = OpenAIEmbeddingProvider(
                model_name=self.config.openai_model,
                api_key=self.config.openai_api_key,
            )
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")
        
        logger.info(
            f"EmbeddingService initialized with {provider} provider, "
            f"dimension={self._provider.dimension}"
        )
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        
        return self._provider.embed_text(text)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        
        if not valid_texts:
            return []
        
        return self._provider.embed_batch(valid_texts)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a user query for retrieval.
        
        This is a semantic alias for embed_text, used for clarity
        when embedding user queries vs documents.
        
        Args:
            query: User's question/query
            
        Returns:
            Query embedding vector
        """
        return self.embed_text(query)
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._provider.dimension
    
    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._provider.model_name
    
    @property
    def provider_name(self) -> str:
        """Return the provider name (local/openai)."""
        return self.config.provider


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    This is a utility function for comparing embeddings.
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
        
    Returns:
        Similarity score between -1 and 1 (1 = identical)
    """
    arr1 = np.array(vec1)
    arr2 = np.array(vec2)
    
    dot_product = np.dot(arr1, arr2)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def compare_embeddings(
    text1: str, 
    text2: str, 
    service: Optional[EmbeddingService] = None
) -> float:
    """
    Compare semantic similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        service: Optional EmbeddingService (creates new if not provided)
        
    Returns:
        Similarity score between 0 and 1
    """
    if service is None:
        service = EmbeddingService()
    
    emb1 = service.embed_text(text1)
    emb2 = service.embed_text(text2)
    
    return cosine_similarity(emb1, emb2)
