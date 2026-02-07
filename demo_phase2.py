"""
Phase 2 Demo Script - RAG Pipeline

This script demonstrates the complete RAG pipeline:
1. Load documents from knowledge base
2. Create embeddings and store in vector store
3. Query the system and get answers with sources

Usage:
    python demo_phase2.py

Requirements:
    - Documents in data/knowledge_base/
    - Ollama running locally (or configure OpenAI API key)
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src import (
    DocumentChunker,
    EmbeddingService,
    VectorStore,
    LLMService,
    RAGChain,
    RAGAgent,
    ConversationMemory,
)
from config.settings import get_settings


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(step: int, text: str):
    """Print a step indicator."""
    print(f"\n[Step {step}] {text}")
    print("-" * 40)


def demo_components():
    """Demonstrate individual RAG components."""
    print_header("Phase 2: Individual Components Demo")
    settings = get_settings()
    
    # Step 1: Document Chunking
    print_step(1, "Document Chunking")
    chunker = DocumentChunker()
    
    kb_path = settings.knowledge_base_dir
    docs = list(kb_path.glob("*.docx"))
    
    if not docs:
        print("❌ No documents found in knowledge base!")
        print(f"   Please add documents to: {kb_path}")
        return False
    
    print(f"📁 Found {len(docs)} documents in knowledge base")
    
    all_chunks = []
    for doc in docs:
        chunks = chunker.process_document(doc)
        all_chunks.extend(chunks)
        print(f"   ✅ {doc.name}: {len(chunks)} chunks")
    
    print(f"\n📊 Total chunks created: {len(all_chunks)}")
    
    # Step 2: Embeddings
    print_step(2, "Embedding Generation")
    embedding_service = EmbeddingService(provider="local")
    print(f"📐 Using model: {embedding_service.model_name}")
    print(f"📐 Embedding dimension: {embedding_service.dimension}")
    
    # Test embedding
    test_embedding = embedding_service.embed_text("What is the bootcamp schedule?")
    print(f"   ✅ Sample embedding generated: {len(test_embedding)} dimensions")
    
    # Step 3: Vector Store
    print_step(3, "Vector Store Setup")
    vector_store = VectorStore(
        embedding_service=embedding_service,
        provider="faiss",
    )
    
    # Add chunks in smaller batches to avoid memory issues
    batch_size = 20
    added = 0
    print(f"   📥 Adding {len(all_chunks)} chunks in batches of {batch_size}...")
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        batch_added = vector_store.add_chunks(batch)
        added += batch_added
        print(f"      Batch {i//batch_size + 1}: Added {batch_added} chunks")
    
    print(f"   ✅ Total added: {added} chunks to vector store")
    print(f"   📊 Total documents in store: {vector_store.count()}")
    
    # Step 4: Test Retrieval
    print_step(4, "Vector Search Test")
    test_query = "How long is the bootcamp program?"
    results = vector_store.search(test_query, top_k=3)
    
    print(f"🔍 Query: '{test_query}'")
    print(f"📄 Retrieved {len(results)} relevant chunks:\n")
    
    for i, result in enumerate(results, 1):
        print(f"  [{i}] Score: {result.score:.3f}")
        print(f"      Source: {result.chunk.source}")
        print(f"      Preview: {result.chunk.text[:150]}...")
        print()
    
    # Step 5: LLM Service Info
    print_step(5, "LLM Service")
    print(f"🤖 Default provider: {settings.llm.provider}")
    print(f"   Ollama URL: {settings.llm.ollama_base_url}")
    print(f"   Ollama model: {settings.llm.ollama_model}")
    print("\n   💡 To test LLM generation, ensure Ollama is running:")
    print("      ollama serve")
    print("      ollama pull llama2")
    
    return True


def demo_rag_agent():
    """Demonstrate the complete RAG Agent API."""
    print_header("Phase 2: RAG Agent Demo")
    
    # Use provider from settings (configured via .env)
    settings = get_settings()
    
    # Initialize the agent
    print("\n📦 Initializing RAG Agent...")
    agent = RAGAgent(
        embedding_provider="local",
        llm_provider=settings.llm.provider,  # Use configured provider from .env
        vector_store_provider="faiss",
    )
    
    # Get stats
    stats = agent.get_stats()
    print(f"\n📊 Agent Statistics:")
    print(f"   Embedding: {stats['embedding']['provider']} ({stats['embedding']['model']})")
    print(f"   LLM: {stats['llm']['provider']} ({stats['llm']['model']})")
    print(f"   Vector Store: {stats['knowledge_base']['provider']}")
    print(f"   Documents: {stats['knowledge_base']['total_chunks']} chunks")
    
    # Ingest knowledge base
    print("\n📥 Ingesting knowledge base documents...")
    result = agent.ingest_directory("data/knowledge_base")
    
    if result["success"]:
        print(f"   ✅ Processed {result['documents_processed']} documents")
        print(f"   📊 Total chunks: {result['total_chunks']}")
    else:
        print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
        return
    
    # Test queries
    test_questions = [
        "How long is the AI Bootcamp program?",
        "What happens during Week 1 of the bootcamp?",
        "What are the award tiers for interns?",
        "How does the team matching process work?",
    ]
    
    print("\n" + "=" * 60)
    print("  🎯 Testing RAG Queries")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[Q{i}] {question}")
        print("-" * 50)
        
        try:
            # Query the agent
            response = agent.query(question)
            
            print(f"📝 Answer: {response['answer'][:500]}...")
            print(f"\n📈 Confidence: {response['confidence']:.2%}")
            
            if response['sources']:
                print("📚 Sources:")
                for source in response['sources'][:3]:
                    print(f"   - {source['source']} (score: {source['score']:.3f})")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("   💡 Make sure Ollama is running with: ollama serve")
    
    # Test conversation memory
    print("\n" + "=" * 60)
    print("  💬 Testing Conversation Memory")
    print("=" * 60)
    
    conv_id = "demo_user"
    
    questions = [
        "What is RAG?",
        "How is it used in the bootcamp projects?",
    ]
    
    for q in questions:
        print(f"\n👤 User: {q}")
        try:
            response = agent.query(q, conversation_id=conv_id)
            print(f"🤖 Bot: {response['answer'][:300]}...")
        except Exception as e:
            print(f"❌ Error: {e}")


def demo_similarity_search():
    """Demonstrate similarity search without LLM."""
    print_header("Phase 2: Similarity Search Demo")
    
    agent = RAGAgent(
        embedding_provider="local",
        vector_store_provider="faiss",
    )
    
    # Ingest documents
    agent.ingest_directory("data/knowledge_base")
    
    # Search without generating
    print("\n🔍 Searching for similar content (no LLM required)...\n")
    
    queries = [
        "bootcamp schedule",
        "team matching",
        "RAG implementation",
    ]
    
    for query in queries:
        print(f"Query: '{query}'")
        print("-" * 40)
        
        results = agent.search_similar(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  [{i}] Score: {result['score']:.3f}")
            print(f"      Source: {result['source']}")
            print(f"      Text: {result['text'][:100]}...")
            print()


def main():
    """Run the Phase 2 demo."""
    print("\n" + "🚀" * 25)
    print("\n  Discord RAG FAQ Chatbot - Phase 2 Demo")
    print("  Testing the complete RAG Pipeline")
    print("\n" + "🚀" * 25)
    
    # Check for documents
    settings = get_settings()
    kb_path = settings.knowledge_base_dir
    
    if not kb_path.exists():
        kb_path.mkdir(parents=True, exist_ok=True)
        print(f"\n⚠️  Knowledge base directory created: {kb_path}")
        print("    Please add your documents and run again.")
        return
    
    docs = list(kb_path.glob("*.*"))
    if not docs:
        print(f"\n⚠️  No documents found in: {kb_path}")
        print("    Please add .pdf, .docx, .txt, or .md files.")
        return
    
    # Run demos
    try:
        # Demo 1: Individual components
        success = demo_components()
        
        if not success:
            return
        
        # Demo 2: Similarity search (no LLM needed)
        demo_similarity_search()
        
        # Demo 3: Full RAG (requires LLM)
        print("\n" + "=" * 60)
        print("  ⚠️  Full RAG Demo requires Ollama or OpenAI API")
        print("=" * 60)
        print("\nTo test the full RAG pipeline with LLM generation:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Run: ollama serve")
        print("3. Pull a model: ollama pull llama2")
        print("4. Run this demo again")
        print("\nOr set OPENAI_API_KEY in .env to use OpenAI.")
        
        user_input = input("\nWould you like to try the full RAG demo? (y/n): ")
        
        if user_input.lower() == 'y':
            demo_rag_agent()
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("  ✅ Phase 2 Demo Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run tests: pytest tests/ -v")
    print("  2. Set up Ollama for local LLM inference")
    print("  3. Integrate with Discord bot (Phase 3)")
    print()


if __name__ == "__main__":
    main()
