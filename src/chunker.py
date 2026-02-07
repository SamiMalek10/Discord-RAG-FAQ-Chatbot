"""
Document Chunker Module

Handles document segmentation for FAQ/training materials with metadata preservation.
Uses LangChain for chunking (as permitted by assignment) but understands each step.

Chunking Strategy:
- Recursive Character Splitting: Splits on natural boundaries (paragraphs, sentences)
- Target size: 200-500 tokens per chunk
- Overlap: 50-100 tokens for context preservation
- Metadata: Source document, section headers, chunk index
"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime

# LangChain imports (allowed for chunking per assignment)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

from config.settings import get_settings, ChunkingConfig

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """
    Represents a single chunk of text with metadata.
    
    Attributes:
        text: The actual text content of the chunk
        chunk_id: Unique identifier for this chunk
        source: Original document filename/path
        chunk_index: Position of this chunk in the document (0-indexed)
        total_chunks: Total number of chunks from this document
        metadata: Additional metadata (section headers, page numbers, etc.)
        embedding: Vector embedding (populated later by EmbeddingService)
    """
    
    text: str
    chunk_id: str
    source: str
    chunk_index: int
    total_chunks: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Generate chunk_id if not provided."""
        if not self.chunk_id:
            # Create deterministic ID from content + source + index
            content_hash = hashlib.md5(
                f"{self.source}:{self.chunk_index}:{self.text[:100]}".encode()
            ).hexdigest()[:12]
            self.chunk_id = f"{Path(self.source).stem}_{self.chunk_index}_{content_hash}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage."""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create Chunk from dictionary."""
        return cls(
            text=data["text"],
            chunk_id=data["chunk_id"],
            source=data["source"],
            chunk_index=data["chunk_index"],
            total_chunks=data.get("total_chunks", 0),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
        )


class DocumentChunker:
    """
    Handles document loading and chunking with metadata preservation.
    
    Supports multiple file formats:
    - PDF (.pdf)
    - Plain text (.txt, .md)
    - Word documents (.docx)
    
    Chunking Process:
    1. Load document using appropriate loader
    2. Extract text and metadata
    3. Split into chunks using recursive character splitting
    4. Preserve metadata (source, page numbers, section headers)
    5. Generate unique chunk IDs
    
    Example:
        chunker = DocumentChunker()
        chunks = chunker.process_document("path/to/document.pdf")
        for chunk in chunks:
            print(f"Chunk {chunk.chunk_index}: {chunk.text[:100]}...")
    """
    
    # Supported file extensions and their loaders
    SUPPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".docx": Docx2txtLoader,
    }
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        config: Optional[ChunkingConfig] = None,
    ):
        """
        Initialize the DocumentChunker.
        
        Args:
            chunk_size: Target characters per chunk (default from config)
            chunk_overlap: Overlap between chunks (default from config)
            config: Optional ChunkingConfig instance
        """
        settings = get_settings()
        self.config = config or settings.chunking
        
        self.chunk_size = chunk_size or self.config.chunk_size
        self.chunk_overlap = chunk_overlap or self.config.chunk_overlap
        
        # Initialize the text splitter
        # Using recursive splitting on natural boundaries
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraph breaks (highest priority)
                "\n",    # Line breaks
                ". ",    # Sentences
                "? ",    # Questions
                "! ",    # Exclamations
                "; ",    # Semicolons
                ", ",    # Commas
                " ",     # Words
                "",      # Characters (last resort)
            ],
            keep_separator=True,
        )
        
        logger.info(
            f"DocumentChunker initialized: chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )
    
    def _get_loader(self, file_path: Path):
        """
        Get the appropriate document loader for the file type.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Document loader instance
            
        Raises:
            ValueError: If file type is not supported
        """
        extension = file_path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )
        
        loader_class = self.SUPPORTED_EXTENSIONS[extension]
        return loader_class(str(file_path))
    
    def _extract_metadata(
        self, 
        file_path: Path, 
        page_content: str, 
        page_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract and enrich metadata from document.
        
        Args:
            file_path: Path to the source document
            page_content: Text content of the page/section
            page_metadata: Metadata from the loader
            
        Returns:
            Enriched metadata dictionary
        """
        metadata = {
            "source_file": file_path.name,
            "source_path": str(file_path),
            "file_type": file_path.suffix.lower(),
            "processed_at": datetime.utcnow().isoformat(),
        }
        
        # Add page number if available (from PDF)
        if "page" in page_metadata:
            metadata["page_number"] = page_metadata["page"] + 1  # 1-indexed
        
        # Try to extract section headers (lines starting with # or ALL CAPS)
        lines = page_content.split("\n")
        headers = []
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line.startswith("#"):
                headers.append(line.lstrip("#").strip())
            elif line.isupper() and len(line) > 3 and len(line) < 100:
                headers.append(line)
        
        if headers:
            metadata["section_headers"] = headers
        
        # Add any additional metadata from loader
        for key, value in page_metadata.items():
            if key not in metadata and key != "source":
                metadata[key] = value
        
        return metadata
    
    def process_document(
        self, 
        file_path: str | Path,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Process a single document and return list of chunks.
        
        This is the main entry point for document processing.
        
        Args:
            file_path: Path to the document file
            additional_metadata: Extra metadata to add to all chunks
            
        Returns:
            List of Chunk objects with text and metadata
            
        Raises:
            FileNotFoundError: If document doesn't exist
            ValueError: If file type is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        logger.info(f"Processing document: {file_path.name}")
        
        # Step 1: Load document
        loader = self._get_loader(file_path)
        documents = loader.load()
        
        logger.debug(f"Loaded {len(documents)} pages/sections from {file_path.name}")
        
        # Step 2: Split into chunks
        all_chunks: List[Chunk] = []
        chunk_index = 0
        
        for doc in documents:
            # Extract metadata for this page/section
            page_metadata = self._extract_metadata(
                file_path, 
                doc.page_content, 
                doc.metadata
            )
            
            # Add any additional metadata provided
            if additional_metadata:
                page_metadata.update(additional_metadata)
            
            # Split the page content into chunks
            text_chunks = self._splitter.split_text(doc.page_content)
            
            for text in text_chunks:
                # Skip empty or very short chunks
                if len(text.strip()) < 10:
                    continue
                
                chunk = Chunk(
                    text=text.strip(),
                    chunk_id="",  # Will be auto-generated
                    source=file_path.name,
                    chunk_index=chunk_index,
                    metadata=page_metadata.copy(),
                )
                all_chunks.append(chunk)
                chunk_index += 1
        
        # Update total_chunks count
        total = len(all_chunks)
        for chunk in all_chunks:
            chunk.total_chunks = total
        
        logger.info(
            f"Created {total} chunks from {file_path.name} "
            f"(avg {sum(len(c.text) for c in all_chunks) // max(total, 1)} chars/chunk)"
        )
        
        return all_chunks
    
    def process_directory(
        self,
        directory_path: str | Path,
        recursive: bool = True,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Process all supported documents in a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search subdirectories
            additional_metadata: Extra metadata for all chunks
            
        Returns:
            List of all chunks from all documents
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")
        
        all_chunks: List[Chunk] = []
        
        # Find all supported files
        pattern = "**/*" if recursive else "*"
        
        for extension in self.SUPPORTED_EXTENSIONS.keys():
            for file_path in directory_path.glob(f"{pattern}{extension}"):
                try:
                    chunks = self.process_document(file_path, additional_metadata)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue
        
        logger.info(
            f"Processed directory {directory_path.name}: "
            f"{len(all_chunks)} total chunks"
        )
        
        return all_chunks
    
    def process_text(
        self,
        text: str,
        source_name: str = "direct_input",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Process raw text directly (for testing or API input).
        
        Args:
            text: Raw text to chunk
            source_name: Name to use as source
            metadata: Optional metadata
            
        Returns:
            List of Chunk objects
        """
        metadata = metadata or {}
        metadata["source_type"] = "direct_input"
        metadata["processed_at"] = datetime.utcnow().isoformat()
        
        text_chunks = self._splitter.split_text(text)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text.strip()) < 10:
                continue
                
            chunk = Chunk(
                text=chunk_text.strip(),
                chunk_id="",
                source=source_name,
                chunk_index=i,
                metadata=metadata.copy(),
            )
            chunks.append(chunk)
        
        # Update total
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks


# Convenience function for quick usage
def chunk_document(file_path: str | Path) -> List[Chunk]:
    """
    Quick function to chunk a single document with default settings.
    
    Args:
        file_path: Path to the document
        
    Returns:
        List of Chunk objects
    """
    chunker = DocumentChunker()
    return chunker.process_document(file_path)
