"""
Debug Retrieval Script

Diagnoses why some queries return 0 confidence / no results.
This script will help identify issues with:
1. Chunking - Are documents being chunked properly?
2. Embeddings - Are embeddings being generated correctly?
3. Similarity scores - What scores are queries getting?
4. Threshold - Is the threshold too high?
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.chunker import DocumentChunker
from src.embeddings import EmbeddingService, cosine_similarity
from src.vector_store import VectorStore
from config.settings import get_settings
import numpy as np


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def debug_chunking():
    """Debug document chunking."""
    print_section("1. CHUNKING ANALYSIS")
    
    settings = get_settings()
    chunker = DocumentChunker()
    kb_path = settings.knowledge_base_dir
    
    print(f"\n📁 Knowledge base path: {kb_path}")
    print(f"📐 Chunk size: {settings.chunking.chunk_size}")
    print(f"📐 Chunk overlap: {settings.chunking.chunk_overlap}")
    
    docs = list(kb_path.glob("*.docx")) + list(kb_path.glob("*.pdf")) + list(kb_path.glob("*.txt")) + list(kb_path.glob("*.md"))
    print(f"\n📄 Found {len(docs)} documents:")
    
    all_chunks = []
    for doc in docs:
        print(f"\n  📄 {doc.name}")
        chunks = chunker.process_document(doc)
        all_chunks.extend(chunks)
        
        print(f"     Chunks created: {len(chunks)}")
        
        if chunks:
            # Sample chunk analysis
            avg_len = sum(len(c.text) for c in chunks) / len(chunks)
            min_len = min(len(c.text) for c in chunks)
            max_len = max(len(c.text) for c in chunks)
            
            print(f"     Avg chunk length: {avg_len:.0f} chars")
            print(f"     Min/Max length: {min_len}/{max_len} chars")
            
            # Show first chunk sample
            print(f"     First chunk preview: '{chunks[0].text[:100]}...'")
    
    print(f"\n📊 TOTAL CHUNKS: {len(all_chunks)}")
    return all_chunks


def debug_embeddings(chunks):
    """Debug embedding generation."""
    print_section("2. EMBEDDING ANALYSIS")
    
    embedding_service = EmbeddingService(provider="local")
    print(f"\n🔢 Model: {embedding_service.model_name}")
    print(f"🔢 Dimension: {embedding_service.dimension}")
    
    # Test with sample queries
    test_queries = [
        "How long is the bootcamp?",
        "What is the schedule?",
        "award tiers",
        "team matching process",
        "Week 1",
        "RAG implementation",
    ]
    
    print("\n📊 Query Embedding Samples:")
    query_embeddings = {}
    for query in test_queries:
        emb = embedding_service.embed_query(query)
        query_embeddings[query] = emb
        print(f"   '{query}': dim={len(emb)}, norm={np.linalg.norm(emb):.4f}")
    
    return embedding_service, query_embeddings


def debug_similarity(chunks, embedding_service, test_queries):
    """Debug similarity scores between queries and chunks."""
    print_section("3. SIMILARITY SCORE ANALYSIS")
    
    settings = get_settings()
    print(f"\n⚙️  Current similarity threshold: {settings.retrieval.similarity_threshold}")
    print(f"⚙️  Top-K setting: {settings.retrieval.top_k}")
    
    # Embed a subset of chunks for analysis
    sample_chunks = chunks[:30] if len(chunks) > 30 else chunks
    print(f"\n📊 Analyzing similarity for {len(sample_chunks)} sample chunks...")
    
    # Embed chunks
    chunk_texts = [c.text for c in sample_chunks]
    chunk_embeddings = embedding_service.embed_batch(chunk_texts)
    
    # Test each query
    for query, query_emb in test_queries.items():
        print(f"\n🔍 Query: '{query}'")
        
        # Calculate similarities
        similarities = []
        for i, chunk_emb in enumerate(chunk_embeddings):
            sim = cosine_similarity(query_emb, chunk_emb)
            similarities.append((sim, sample_chunks[i]))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Show top 5
        print(f"   Top 5 matches:")
        for i, (sim, chunk) in enumerate(similarities[:5]):
            above_threshold = "✅" if sim >= settings.retrieval.similarity_threshold else "❌"
            print(f"   {above_threshold} [{i+1}] Score: {sim:.4f} | Source: {chunk.source[:30]}")
            print(f"         Text: '{chunk.text[:80]}...'")
        
        # Count how many pass threshold
        passing = sum(1 for sim, _ in similarities if sim >= settings.retrieval.similarity_threshold)
        print(f"   📊 Chunks above threshold ({settings.retrieval.similarity_threshold}): {passing}/{len(similarities)}")


def debug_vector_store():
    """Debug vector store retrieval."""
    print_section("4. VECTOR STORE RETRIEVAL TEST")
    
    settings = get_settings()
    embedding_service = EmbeddingService(provider="local")
    vector_store = VectorStore(embedding_service=embedding_service, provider="faiss")
    
    # Load documents
    chunker = DocumentChunker()
    kb_path = settings.knowledge_base_dir
    docs = list(kb_path.glob("*.*"))
    
    all_chunks = []
    for doc in docs:
        if doc.suffix.lower() in ['.docx', '.pdf', '.txt', '.md']:
            chunks = chunker.process_document(doc)
            all_chunks.extend(chunks)
    
    # Add to vector store in batches
    print(f"\n📥 Adding {len(all_chunks)} chunks to vector store...")
    batch_size = 20
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        vector_store.add_chunks(batch)
    
    print(f"   ✅ Vector store count: {vector_store.count()}")
    
    # Test queries with different thresholds
    test_queries = [
        "How long is the bootcamp program?",
        "What happens in Week 1?",
        "What are the award tiers?",
        "How does team matching work?",
        "What is RAG?",
        "bootcamp schedule",
    ]
    
    thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("\n📊 Retrieval results at different thresholds:\n")
    
    for query in test_queries:
        print(f"🔍 '{query}'")
        
        for threshold in thresholds_to_test:
            results = vector_store.search(query, top_k=5, threshold=threshold)
            top_score = results[0].score if results else 0
            print(f"   threshold={threshold}: {len(results)} results (top score: {top_score:.3f})")
        
        # Show actual results at current threshold
        results = vector_store.search(query, top_k=3, threshold=settings.retrieval.similarity_threshold)
        if results:
            print(f"   📄 Current threshold ({settings.retrieval.similarity_threshold}) results:")
            for r in results[:2]:
                print(f"      - [{r.score:.3f}] {r.chunk.source}: '{r.chunk.text[:60]}...'")
        else:
            print(f"   ⚠️  NO RESULTS at current threshold!")
        print()


def recommend_settings():
    """Provide recommendations based on analysis."""
    print_section("5. RECOMMENDATIONS")
    
    settings = get_settings()
    
    print("""
Based on typical RAG systems for FAQ documents:

📌 CURRENT SETTINGS:
""")
    print(f"   similarity_threshold: {settings.retrieval.similarity_threshold}")
    print(f"   top_k: {settings.retrieval.top_k}")
    print(f"   chunk_size: {settings.chunking.chunk_size}")
    print(f"   chunk_overlap: {settings.chunking.chunk_overlap}")
    
    print("""
📌 RECOMMENDED ADJUSTMENTS:

1. SIMILARITY THRESHOLD:
   - Current: {threshold}
   - Recommended: 0.2 - 0.4 for sentence-transformers
   - Why: MiniLM embeddings typically have lower cosine similarity 
         scores than OpenAI embeddings

2. CHUNK SIZE:
   - Current: {chunk_size}
   - Recommended: 300-500 for FAQ documents
   - Why: Smaller chunks = more precise retrieval for Q&A

3. CHUNK OVERLAP:
   - Current: {overlap}
   - Recommended: 50-100 (10-20% of chunk size)
   - Why: Preserves context across chunk boundaries

4. TOP-K:
   - Current: {top_k}
   - Recommended: 3-5 for FAQ
   - Why: More context helps but too much can confuse LLM

To apply recommended settings, update your .env file or config/settings.py
""".format(
        threshold=settings.retrieval.similarity_threshold,
        chunk_size=settings.chunking.chunk_size,
        overlap=settings.chunking.chunk_overlap,
        top_k=settings.retrieval.top_k,
    ))


def main():
    print("\n" + "🔍" * 30)
    print("\n  RAG RETRIEVAL DEBUG TOOL")
    print("\n" + "🔍" * 30)
    
    try:
        # Step 1: Analyze chunking
        chunks = debug_chunking()
        
        if not chunks:
            print("\n❌ No chunks created! Check your knowledge base documents.")
            return
        
        # Step 2: Analyze embeddings
        embedding_service, query_embeddings = debug_embeddings(chunks)
        
        # Step 3: Analyze similarity scores
        debug_similarity(chunks, embedding_service, query_embeddings)
        
        # Step 4: Test vector store retrieval
        debug_vector_store()
        
        # Step 5: Recommendations
        recommend_settings()
        
    except Exception as e:
        print(f"\n❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("  Debug complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
