#!/usr/bin/env python3
"""
MongoDB Atlas Demo

Demonstrates the MongoDB Vector Store functionality.
Shows how to:
1. Connect to MongoDB Atlas
2. Add documents with embeddings
3. Perform vector search
4. Use filtering

Usage:
    python scripts/demo_mongodb.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO)


def demo_mongodb():
    """Demonstrate MongoDB Vector Store functionality."""
    print("\n" + "=" * 60)
    print("  🍃 MongoDB Atlas Vector Store Demo")
    print("=" * 60 + "\n")
    
    # Check configuration
    mongodb_uri = os.getenv("MONGODB_URI", "")
    if not mongodb_uri or "username:password" in mongodb_uri:
        print("❌ MongoDB URI not configured!")
        print("\nPlease set MONGODB_URI in your .env file.")
        print("Run: python scripts/setup_mongodb.py")
        return
    
    provider = os.getenv("VECTOR_STORE_PROVIDER", "faiss")
    if provider != "mongodb":
        print(f"⚠️  Current provider is '{provider}', switching to MongoDB for demo...\n")
    
    # Import after env is loaded
    from src.embeddings import EmbeddingService
    from src.vector_store import MongoDBVectorStore, VectorStore
    from src.chunker import Chunk
    
    print("📦 Initializing services...")
    
    # Create embedding service
    embedding_service = EmbeddingService()
    print(f"   Embedding model: {embedding_service.model_name}")
    print(f"   Dimension: {embedding_service.dimension}")
    
    # Create MongoDB store directly
    print("\n🍃 Connecting to MongoDB Atlas...")
    
    try:
        store = MongoDBVectorStore(
            dimension=embedding_service.dimension,
            uri=mongodb_uri,
            database=os.getenv("MONGODB_DATABASE", "rag_chatbot"),
            collection=os.getenv("MONGODB_COLLECTION", "knowledge_base"),
            vector_index=os.getenv("MONGODB_VECTOR_INDEX", "vector_index"),
        )
        store._connect()
        print("✅ Connected successfully!")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Check current count
    current_count = store.count()
    print(f"\n📊 Current document count: {current_count}")
    
    # Demo: Add sample chunks
    print("\n" + "-" * 40)
    print("Demo: Adding sample chunks")
    print("-" * 40)
    
    sample_texts = [
        "The AI Bootcamp runs for 3 weeks with hands-on projects.",
        "Week 1 focuses on RAG fundamentals and document processing.",
        "Week 2 covers embeddings and vector databases like MongoDB Atlas.",
        "Week 3 is dedicated to Discord bot integration and deployment.",
        "MongoDB Atlas provides scalable vector search capabilities.",
    ]
    
    # Create chunks with embeddings
    chunks = []
    for i, text in enumerate(sample_texts):
        embedding = embedding_service.embed_query(text)
        chunk = Chunk(
            text=text,
            chunk_id=f"demo_chunk_{i}",
            source="demo.txt",
            chunk_index=i,
            total_chunks=len(sample_texts),
            embedding=embedding,
        )
        chunks.append(chunk)
    
    print(f"Created {len(chunks)} chunks with embeddings")
    
    # Add to MongoDB
    added = store.add_chunks(chunks)
    print(f"✅ Added {added} chunks to MongoDB")
    
    # Check new count
    new_count = store.count()
    print(f"New document count: {new_count}")
    
    # Demo: Vector search
    print("\n" + "-" * 40)
    print("Demo: Vector Search")
    print("-" * 40)
    
    queries = [
        "What topics are covered in Week 2?",
        "How long is the bootcamp?",
        "What database does the project use?",
    ]
    
    for query in queries:
        print(f"\n🔍 Query: \"{query}\"")
        
        # Embed the query
        query_embedding = embedding_service.embed_query(query)
        
        # Search
        results = store.search(
            query_embedding=query_embedding,
            top_k=3,
            threshold=0.3,
        )
        
        if results:
            print(f"   Found {len(results)} results:")
            for r in results:
                print(f"   • [{r.score:.3f}] {r.chunk.text[:60]}...")
        else:
            print("   No results found (check vector index)")
    
    # Demo: Cleanup (optional)
    print("\n" + "-" * 40)
    print("Demo: Cleanup")
    print("-" * 40)
    
    response = input("Delete demo chunks? [y/N]: ").strip().lower()
    if response == 'y':
        demo_ids = [f"demo_chunk_{i}" for i in range(len(sample_texts))]
        deleted = store.delete(demo_ids)
        print(f"✅ Deleted {deleted} demo chunks")
    
    # Summary
    print("\n" + "=" * 60)
    print("  ✅ MongoDB Demo Complete!")
    print("=" * 60)
    print(f"""
MongoDB Atlas is working correctly!

Configuration:
  URI: {mongodb_uri[:30]}...
  Database: {store.database_name}
  Collection: {store.collection_name}
  Index: {store.vector_index}
  Documents: {store.count()}

To use MongoDB in your bot:
  1. Set VECTOR_STORE_PROVIDER=mongodb in .env
  2. Run: python run_bot.py
""")


if __name__ == "__main__":
    demo_mongodb()
