"""
Configuration settings for the RAG Chatbot.

This module handles all configuration management using environment variables.
No hardcoded values - everything is configurable via .env file.
"""

import os
from dataclasses import dataclass, field
from typing import Literal, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    
    provider: Literal["local", "openai"] = "local"
    local_model: str = "all-MiniLM-L6-v2"
    openai_model: str = "text-embedding-3-small"
    openai_api_key: Optional[str] = None
    
    # Embedding dimensions (depends on model)
    # all-MiniLM-L6-v2: 384
    # text-embedding-3-small: 1536
    @property
    def dimension(self) -> int:
        """Return embedding dimension based on selected model."""
        if self.provider == "local":
            model_dimensions = {
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
                "paraphrase-MiniLM-L6-v2": 384,
            }
            return model_dimensions.get(self.local_model, 384)
        else:
            model_dimensions = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            }
            return model_dimensions.get(self.openai_model, 1536)


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    
    provider: Literal["ollama", "openai", "gemini", "mistral"] = "ollama"
    
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"
    
    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    
    # Gemini settings
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-pro"
    
    # Mistral settings
    mistral_api_key: Optional[str] = None
    mistral_model: str = "mistral-small-latest"


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    
    provider: Literal["faiss", "mongodb"] = "faiss"
    
    # MongoDB settings
    mongodb_uri: Optional[str] = None
    mongodb_database: str = "rag_chatbot"
    mongodb_collection: str = "knowledge_base"
    mongodb_vector_index: str = "vector_index"
    
    # FAISS settings
    faiss_index_path: str = "./data/faiss_index"


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    
    chunk_size: int = 400  # Target tokens per chunk (200-500 recommended)
    chunk_overlap: int = 50  # Overlap between chunks for context preservation
    
    # Chunking strategy: "recursive" or "semantic"
    strategy: Literal["recursive", "semantic"] = "recursive"


@dataclass
class RetrievalConfig:
    """Configuration for retrieval settings."""
    
    top_k: int = 5  # Number of chunks to retrieve
    similarity_threshold: float = 0.3  # Minimum similarity score (lower = more results)
    max_context_length: int = 4000  # Max tokens for LLM context


@dataclass
class Settings:
    """
    Main settings class that aggregates all configurations.
    
    Usage:
        settings = get_settings()
        print(settings.embedding.provider)
        print(settings.llm.ollama_model)
    """
    
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    knowledge_base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "knowledge_base")
    
    # Logging
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "Settings":
        """
        Create Settings instance from environment variables.
        
        This is the primary way to instantiate Settings.
        """
        embedding = EmbeddingConfig(
            provider=os.getenv("EMBEDDING_PROVIDER", "local"),  # type: ignore
            local_model=os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            openai_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        
        llm = LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "ollama"),  # type: ignore
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama2"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_LLM_MODEL", "gpt-3.5-turbo"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-pro"),
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            mistral_model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        )
        
        vector_store = VectorStoreConfig(
            provider=os.getenv("VECTOR_STORE_PROVIDER", "faiss"),  # type: ignore
            mongodb_uri=os.getenv("MONGODB_URI"),
            mongodb_database=os.getenv("MONGODB_DATABASE", "rag_chatbot"),
            mongodb_collection=os.getenv("MONGODB_COLLECTION", "knowledge_base"),
            mongodb_vector_index=os.getenv("MONGODB_VECTOR_INDEX", "vector_index"),
            faiss_index_path=os.getenv("FAISS_INDEX_PATH", "./data/faiss_index"),
        )
        
        chunking = ChunkingConfig(
            chunk_size=int(os.getenv("CHUNK_SIZE", "400")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        )
        
        retrieval = RetrievalConfig(
            top_k=int(os.getenv("TOP_K_RESULTS", "5")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.3")),  # Lower threshold for MiniLM
            max_context_length=int(os.getenv("MAX_CONTEXT_LENGTH", "4000")),
        )
        
        return cls(
            embedding=embedding,
            llm=llm,
            vector_store=vector_store,
            chunking=chunking,
            retrieval=retrieval,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


# Singleton pattern for settings
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the singleton Settings instance.
    
    Returns:
        Settings: The application settings loaded from environment.
    """
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings from environment (useful for testing).
    
    Returns:
        Settings: Fresh settings instance.
    """
    global _settings
    load_dotenv(override=True)
    _settings = Settings.from_env()
    return _settings
