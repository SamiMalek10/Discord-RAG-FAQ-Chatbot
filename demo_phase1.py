"""
Quick Demo Script - Phase 1 Verification

This script demonstrates the Phase 1 components:
1. DocumentChunker - Document loading and chunking
2. EmbeddingService - Vector embedding generation  
3. VectorStore - FAISS-based vector storage and search

Run: python demo_phase1.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("=" * 60)
    print("Discord RAG FAQ Chatbot - Phase 1 Demo")
    print("=" * 60)
    
    # Step 1: Test Configuration
    print("\n📋 Step 1: Loading Configuration...")
    try:
        from config.settings import get_settings
        settings = get_settings()
        print(f"   ✅ Embedding Provider: {settings.embedding.provider}")
        print(f"   ✅ Embedding Model: {settings.embedding.local_model}")
        print(f"   ✅ Vector Store: {settings.vector_store.provider}")
        print(f"   ✅ Chunk Size: {settings.chunking.chunk_size}")
    except Exception as e:
        print(f"   ❌ Configuration Error: {e}")
        return
    
    # Step 2: Test Document Chunker
    print("\n📄 Step 2: Testing DocumentChunker...")
    try:
        from src.chunker import DocumentChunker
        
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=30)
        
        # Test with sample text
        sample_text = """
        # AI Bootcamp FAQ
        
        ## What is the bootcamp schedule?
        
        The AI Bootcamp runs for 12 weeks, with sessions every Monday and Wednesday.
        Each session is 3 hours long, from 6 PM to 9 PM.
        
        ## What will I learn?
        
        You will learn:
        - Python programming fundamentals
        - Machine learning basics
        - Deep learning with PyTorch
        - Natural Language Processing
        - Computer Vision
        - Deployment and MLOps
        
        ## What are the prerequisites?
        
        Basic programming knowledge is required. Familiarity with Python is helpful
        but not mandatory. We will cover Python basics in the first two weeks.
        """
        
        chunks = chunker.process_text(sample_text, source_name="sample_faq.md")
        
        print(f"   ✅ Created {len(chunks)} chunks from sample text")
        print(f"   ✅ First chunk preview: '{chunks[0].text[:50]}...'")
        print(f"   ✅ Chunk IDs generated: {chunks[0].chunk_id}")
    except Exception as e:
        print(f"   ❌ Chunker Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Test Embedding Service
    print("\n🔢 Step 3: Testing EmbeddingService...")
    try:
        from src.embeddings import EmbeddingService, cosine_similarity
        
        print("   ⏳ Loading embedding model (first time may take a moment)...")
        embedding_service = EmbeddingService(provider="local")
        
        # Test single embedding
        test_text = "What is the bootcamp schedule?"
        embedding = embedding_service.embed_text(test_text)
        
        print(f"   ✅ Model: {embedding_service.model_name}")
        print(f"   ✅ Embedding Dimension: {embedding_service.dimension}")
        print(f"   ✅ Generated embedding with {len(embedding)} dimensions")
        
        # Test similarity
        text1 = "What is the schedule for the bootcamp?"
        text2 = "When does the bootcamp meet?"
        text3 = "I like pizza and ice cream."
        
        emb1 = embedding_service.embed_text(text1)
        emb2 = embedding_service.embed_text(text2)
        emb3 = embedding_service.embed_text(text3)
        
        sim_similar = cosine_similarity(emb1, emb2)
        sim_different = cosine_similarity(emb1, emb3)
        
        print(f"   ✅ Similarity (similar questions): {sim_similar:.4f}")
        print(f"   ✅ Similarity (different topics): {sim_different:.4f}")
        
    except ImportError as e:
        print(f"   ⚠️ Embedding requires dependencies: {e}")
        print("   💡 Run: pip install sentence-transformers")
        embedding_service = None
    except Exception as e:
        print(f"   ❌ Embedding Error: {e}")
        import traceback
        traceback.print_exc()
        embedding_service = None
    
    # Step 4: Test Vector Store (FAISS)
    print("\n🗄️ Step 4: Testing VectorStore (FAISS)...")
    if embedding_service:
        try:
            from src.vector_store import VectorStore
            
            # Create vector store
            store = VectorStore(
                embedding_service=embedding_service,
                provider="faiss"
            )
            
            # Add chunks (will auto-embed)
            print("   ⏳ Adding chunks to vector store...")
            added = store.add_chunks(chunks)
            print(f"   ✅ Added {added} chunks to FAISS index")
            print(f"   ✅ Total chunks in store: {store.count()}")
            
            # Test search
            query = "What is the bootcamp schedule?"
            print(f"\n   🔍 Searching for: '{query}'")
            results = store.search(query, top_k=3)
            
            print(f"   ✅ Found {len(results)} results:")
            for result in results:
                preview = result.chunk.text[:60].replace('\n', ' ')
                print(f"      #{result.rank}: (score: {result.score:.4f}) '{preview}...'")
            
        except ImportError as e:
            print(f"   ⚠️ Vector store requires FAISS: {e}")
            print("   💡 Run: pip install faiss-cpu")
        except Exception as e:
            print(f"   ❌ Vector Store Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("   ⏭️ Skipping (embedding service not available)")
    
    # Summary
    print("\n" + "=" * 60)
    print("Phase 1 Demo Complete!")
    print("=" * 60)
    print("""
Next Steps:
1. Add your knowledge base documents to: data/knowledge_base/
2. Install dependencies: pip install -r requirements.txt
3. Configure .env file with your settings
4. Run tests: pytest tests/ -v

Phase 2 will add:
- RAG Chain (retrieval + generation)
- LLM Integration (Ollama/OpenAI)
- RAG Agent API for Backend Engineers
""")


if __name__ == "__main__":
    main()
