"""
Response Quality Test Script

Tests the RAG pipeline response quality after improvements:
1. Better prompts
2. Deduplication
3. Increased context retrieval
"""

import sys
from config.settings import get_settings, reload_settings

# Reload settings to get fresh values
reload_settings()
settings = get_settings()

print("=" * 70)
print("  RAG RESPONSE QUALITY TEST")
print("=" * 70)
print(f"\n📊 Configuration:")
print(f"   Embedding: {settings.embedding.provider} ({settings.embedding.local_model})")
print(f"   LLM: {settings.llm.provider} ({settings.llm.mistral_model})")
print(f"   Vector Store: {settings.vector_store.provider}")
print(f"   Top-K: {settings.retrieval.top_k}")
print(f"   Threshold: {settings.retrieval.similarity_threshold}")
print(f"   Max Context: {settings.retrieval.max_context_length}")

# Initialize RAG Agent
from src.rag_agent import RAGAgent

print("\n📦 Initializing RAG Agent (fresh vector store)...")
agent = RAGAgent(
    embedding_provider='local',
    llm_provider='mistral',
    vector_store_provider='faiss'
)

# Ingest documents
print("\n📥 Ingesting knowledge base...")
stats = agent.ingest_directory('./data/knowledge_base')
print(f"   ✅ Ingested {stats.get('documents_processed', stats.get('total_documents', 0))} documents, {stats.get('chunks_created', stats.get('total_chunks', 0))} chunks")

# Test queries with expected content
test_cases = [
    {
        "query": "How long is the AI Bootcamp program?",
        "expected_keywords": ["week", "3", "duration", "journey"],
        "description": "Duration question"
    },
    {
        "query": "What are all the award tiers for interns?",
        "expected_keywords": ["Trailblazer", "tier", "award", "recognition"],
        "description": "Award tiers question"
    },
    {
        "query": "What happens during Week 1 of the bootcamp?",
        "expected_keywords": ["week 1", "orientation", "foundation", "skills"],
        "description": "Week 1 schedule question"
    },
    {
        "query": "How does the team matching process work?",
        "expected_keywords": ["team", "match", "product manager", "PM", "session"],
        "description": "Team matching question"
    },
    {
        "query": "What is RAG and how is it used in the bootcamp?",
        "expected_keywords": ["RAG", "retrieval", "generation", "knowledge"],
        "description": "Technical RAG question"
    },
]

print("\n" + "=" * 70)
print("  RUNNING QUALITY TESTS")
print("=" * 70)

results = []
for i, test in enumerate(test_cases, 1):
    print(f"\n[Test {i}] {test['description']}")
    print("-" * 60)
    print(f"❓ Question: {test['query']}")
    
    # Query the agent
    response = agent.query(test['query'])
    
    # Check quality
    answer = response['answer']
    confidence = response['confidence']
    sources = response['sources']
    
    # Check if expected keywords are in the answer
    answer_lower = answer.lower()
    found_keywords = [kw for kw in test['expected_keywords'] if kw.lower() in answer_lower]
    keyword_score = len(found_keywords) / len(test['expected_keywords'])
    
    # Is it a rejection ("I don't have information")?
    is_rejection = "don't have" in answer_lower or "no information" in answer_lower
    
    print(f"\n📝 Answer:\n{answer[:500]}{'...' if len(answer) > 500 else ''}")
    print(f"\n📈 Metrics:")
    print(f"   Confidence: {confidence:.1%}")
    print(f"   Sources: {len(sources)}")
    print(f"   Keywords found: {len(found_keywords)}/{len(test['expected_keywords'])} ({keyword_score:.0%})")
    print(f"   Is rejection: {'❌ Yes' if is_rejection else '✅ No'}")
    
    # Quality score
    quality = "✅ GOOD" if keyword_score >= 0.5 and not is_rejection else "⚠️ NEEDS IMPROVEMENT" if keyword_score > 0 else "❌ POOR"
    print(f"   Quality: {quality}")
    
    results.append({
        "test": test['description'],
        "keyword_score": keyword_score,
        "is_rejection": is_rejection,
        "confidence": confidence,
        "quality": quality
    })

# Summary
print("\n" + "=" * 70)
print("  QUALITY SUMMARY")
print("=" * 70)

good_count = sum(1 for r in results if r['quality'] == "✅ GOOD")
avg_confidence = sum(r['confidence'] for r in results) / len(results)
rejection_count = sum(1 for r in results if r['is_rejection'])

print(f"\n📊 Overall Results:")
print(f"   Good responses: {good_count}/{len(results)} ({good_count/len(results):.0%})")
print(f"   Rejections: {rejection_count}/{len(results)}")
print(f"   Average confidence: {avg_confidence:.1%}")

print("\n📋 Individual Results:")
for r in results:
    print(f"   {r['quality']} {r['test']} (conf: {r['confidence']:.1%}, keywords: {r['keyword_score']:.0%})")

if good_count >= 4:
    print("\n🎉 Response quality is GOOD! Ready for Phase 3.")
else:
    print("\n⚠️ Response quality needs improvement. Review the failing tests.")

print("\n" + "=" * 70)
