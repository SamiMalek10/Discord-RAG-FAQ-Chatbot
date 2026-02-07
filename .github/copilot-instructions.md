# GitHub Copilot Instructions: Discord RAG FAQ Chatbot (Data Scientist)

## Project Context
Building a Discord Retrieval-Augmented Generation (RAG) Bot that answers questions based on specific knowledge base documents (AI Bootcamp Journey, Training Materials, Intern FAQ). The bot retrieves relevant context before generating responses to avoid hallucination and provide up-to-date information.

**Phase**: Weeks 1-3 AI Engineering Project  
**Role**: Data Scientist - Core RAG "Intelligence" & Agent Logic  
**Stack**: Python, MongoDB Atlas Vector Search (preferred), Discord.py, Sentence Transformers/OpenAI Embeddings, LLM (Ollama local or cloud API)

---

## Data Scientist Responsibilities (Scope Definition)

When generating code, prioritize these specific assignment requirements:

### Phase 1: Architecture & Research
- **Chunking Strategy**: Implement document segmentation for FAQ/training materials
- **Embedding Models**: Support Sentence Transformers (local) and OpenAI/Cohere (cloud) with comparison logic
- **Vector Store**: MongoDB Atlas Vector Search (primary) or FAISS (local prototyping)
- **LLM Integration**: Support both local (Ollama) and cloud (Gemini/OpenAI) with configurable providers
- **Memory**: Implement conversation context management

### Phase 2: Core Implementation
1. **Data Ingestion Pipeline**:
   - Document chunking (LangChain allowed for chunking)
   - Embedding generation
   - Vector store ingestion

2. **Retrieval Logic**:
   - Query embedding
   - Vector similarity search
   - Top-k retrieval with relevance scoring

3. **RAG Chain**:
   - Prompt construction: `User Query + Retrieved Context`
   - LLM API integration
   - Response generation with source attribution

4. **Evaluation** (if time permits):
   - Relevance/Precision metrics
   - Faithfulness checking (grounding in retrieved context)
   - Answer correctness validation

---

## Technical Constraints & Guidelines

### Vector Database
- **Primary**: MongoDB Atlas Vector Search (aligns with workshop)
- **Fallback**: FAISS for local prototyping only
- **Schema**: Store `text`, `embedding`, `source`, `chunk_id`, `metadata`

### Embedding Models
Support these options with configurable selection:
- **Local/Free**: `all-MiniLM-L6-v2` (Sentence Transformers) or similar
- **Cloud**: OpenAI `text-embedding-3-small` or Cohere
- **Dimension**: Match vector index to model output (384, 768, 1536, etc.)

### LLM Providers
Implement abstraction to support:
- **Local PoC**: Ollama with Llama 2/Mistral
- **Cloud**: OpenAI GPT-3.5/4, Gemini, or Azure DeepSeek R1
- **Requirement**: Must support system prompts and context windows

### Chunking Strategy
- Use **semantic chunking** or **recursive character splitting**
- Target size: 200-500 tokens per chunk
- Preserve context: Overlap of 50-100 tokens between chunks
- Metadata preservation: Source document, section headers, page numbers

---

## Architecture Patterns

### RAG Pipeline Flow
```
User Query (Discord) 
    → Query Embedding 
    → Vector Search (MongoDB) 
    → Retrieve Top-K Chunks 
    → Build Prompt [Context + Question] 
    → LLM Generation 
    → Return Answer with Sources
```

### Component Structure
```python
# Required classes/functions:
class DocumentChunker:
    """Handle PDF/text chunking with metadata preservation"""
    
class EmbeddingService:
    """Abstract embedding generation (local vs cloud)"""
    
class VectorStore:
    """MongoDB Atlas Vector Search interface"""
    
class RAGChain:
    """Orchestrate retrieval + generation"""
    
class RAGEvaluator:
    """Metrics: relevance, faithfulness, correctness"""
```

### API Contract (for Backend Collaboration)
```python
# Expected interface for Backend Engineers
class RAGAgent:
    def query(self, question: str, chat_history: list = None) -> dict:
        """
        Returns: {
            "answer": str,
            "sources": list[dict],
            "confidence": float,
            "retrieved_chunks": list[str]
        }
        """
        pass
    
    def ingest_document(self, file_path: str, metadata: dict) -> bool:
        """Add new documents to knowledge base"""
        pass
```

---

## Code Quality Standards

### Explicit Requirements
- **Type hints** mandatory for all functions
- **Error handling**: Graceful fallbacks for LLM/DB failures
- **Logging**: Structured logs for retrieval metrics, latency, token usage
- **Configuration**: Environment variables for API keys, model names, thresholds
- **No hardcoded paths**: Use relative paths or environment variables

### Prohibited (Anti-Hallucination Rules)
- DO NOT implement chat memory using global variables
- DO NOT hardcode OpenAI as the only LLM option
- DO NOT skip retrieval and send queries directly to LLM
- DO NOT store embeddings in memory for production code (use vector DB)
- DO NOT ignore metadata (sources must be trackable)
- DO NOT use frameworks for the core RAG logic (use "from scratch" approach per assignment, except LangChain for chunking)

### Required Implementations
- Vector search similarity scoring
- Source citation in responses (document name, chunk reference)
- Query preprocessing (lowercasing, punctuation handling optional)
- Context length management (truncate if exceeds LLM limit)

---

## Knowledge Base Documents
Reference these specific source materials:
1. `AI Bootcamp Journey & Learning Path` (schedule information)
2. `Training For AI Engineer Interns` (video links, technical content)
3. `Intern FAQ - AI Bootcamp` (frequently asked questions)

When generating test cases, use questions related to these documents.

---

## Evaluation Metrics (If Implemented)
If generating evaluation code, include:
- **Retrieval Accuracy**: Precision@K of relevant chunks
- **Faithfulness**: LLM-based check if answer is supported by context (no hallucination)
- **Answer Relevance**: Cosine similarity between query and answer embeddings
- **Latency**: End-to-end response time tracking

---

## Dependencies Reference
```txt
pymongo[srv]
sentence-transformers
openai
python-dotenv
langchain (chunking only)
numpy
pandas
pytest (for testing)
discord.py (for context, but DS focuses on RAG core)
```

---

## Workflow Integration
- **Collaboration Point**: Backend Engineers will consume your RAGAgent class via API
- **Deliverables**: Architecture diagram, workflow diagram, working RAG pipeline, evaluation metrics
- **Iteration**: Start simple (in-memory FAISS) → Migrate to MongoDB Atlas → Add evaluation

---

## Remember
1. This is a learning project - prioritize understanding over complex abstractions
2. "From scratch" approach means understand each step (chunking → embedding → vector search → RAG → memory)
3. Document your rationale for model/technology choices in comments
4. Ensure reproducibility: Seed random states, pin dependency versions
```

Save this as `.github/copilot-instructions.md` in your repository root. This ensures Copilot:

1. **Stays grounded** in the actual assignment requirements (Phases 1-2, MongoDB focus)
2. **Respects constraints** (from-scratch RAG logic, specific document sources)
3. **Maintains structure** (type hints, proper abstractions, no globals)
4. **Collaborates correctly** (clear API contracts for Backend Engineers)
5. **Avoids hallucination** (explicit prohibited patterns, required implementations)

The file references your specific knowledge base documents and enforces the "from scratch" approach mentioned in your assignment while allowing LangChain only for chunking as permitted.