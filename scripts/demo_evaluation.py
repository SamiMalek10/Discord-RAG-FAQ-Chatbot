#!/usr/bin/env python3
"""
RAG Evaluation Demo

Demonstrates how to use the RAG Evaluator to measure quality of responses.

Usage:
    python scripts/demo_evaluation.py
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
logger = logging.getLogger(__name__)


def print_banner():
    print("\n" + "=" * 60)
    print("  📊 RAG Evaluation Demo")
    print("  Discord RAG FAQ Chatbot")
    print("=" * 60 + "\n")


def demo_single_evaluation():
    """Demonstrate single response evaluation."""
    print("\n" + "-" * 50)
    print("📝 Demo 1: Single Response Evaluation")
    print("-" * 50)
    
    from src.evaluator import RAGEvaluator
    from src.embeddings import EmbeddingService
    from src.chunker import Chunk
    
    # Initialize
    embedding_service = EmbeddingService()
    evaluator = RAGEvaluator(
        embedding_service=embedding_service,
        use_llm_evaluation=False,  # Faster without LLM
    )
    
    # Create sample chunks (simulating retrieved context)
    chunks = [
        Chunk(
            text="The AI Bootcamp is a 3-week intensive program focused on building practical AI skills.",
            chunk_id="c1",
            source="bootcamp.docx",
            chunk_index=0,
        ),
        Chunk(
            text="Week 1 covers RAG fundamentals including chunking, embeddings, and vector databases.",
            chunk_id="c2",
            source="bootcamp.docx",
            chunk_index=1,
        ),
        Chunk(
            text="Week 2 focuses on LLM integration and building the RAG pipeline from scratch.",
            chunk_id="c3",
            source="bootcamp.docx",
            chunk_index=2,
        ),
    ]
    
    # Test case 1: Good response (grounded in context)
    print("\n✅ Test 1: Good Response (Faithful to Context)")
    result1 = evaluator.evaluate(
        question="How long is the AI Bootcamp?",
        answer="The AI Bootcamp is a 3-week intensive program focused on building practical AI skills.",
        retrieved_chunks=chunks,
    )
    print(f"   Question: How long is the AI Bootcamp?")
    print(f"   Answer: The AI Bootcamp is a 3-week intensive program...")
    print(f"   Overall Score: {result1.overall_score:.2f} ({result1.level.value})")
    print(f"   Faithfulness: {result1.generation_metrics.faithfulness:.2f}")
    print(f"   Answer Relevance: {result1.generation_metrics.answer_relevance:.2f}")
    print(f"   Passed: {result1.passed}")
    
    # Test case 2: Hallucinated response
    print("\n❌ Test 2: Hallucinated Response (Not in Context)")
    result2 = evaluator.evaluate(
        question="How long is the AI Bootcamp?",
        answer="The bootcamp is 6 months long and includes a certification from Google.",
        retrieved_chunks=chunks,
    )
    print(f"   Question: How long is the AI Bootcamp?")
    print(f"   Answer: The bootcamp is 6 months long and includes a certification...")
    print(f"   Overall Score: {result2.overall_score:.2f} ({result2.level.value})")
    print(f"   Faithfulness: {result2.generation_metrics.faithfulness:.2f}")
    print(f"   Answer Relevance: {result2.generation_metrics.answer_relevance:.2f}")
    print(f"   Passed: {result2.passed}")
    
    # Test case 3: "I don't know" response
    print("\n⚠️  Test 3: 'I Don't Know' Response")
    result3 = evaluator.evaluate(
        question="What is the salary for AI engineers?",
        answer="I don't have information about salaries in my knowledge base.",
        retrieved_chunks=chunks,
    )
    print(f"   Question: What is the salary for AI engineers?")
    print(f"   Answer: I don't have information about salaries...")
    print(f"   Overall Score: {result3.overall_score:.2f} ({result3.level.value})")
    print(f"   Faithfulness: {result3.generation_metrics.faithfulness:.2f}")
    print(f"   Coherence: {result3.generation_metrics.coherence:.2f}")
    print(f"   Passed: {result3.passed}")


def demo_batch_evaluation():
    """Demonstrate batch evaluation."""
    print("\n" + "-" * 50)
    print("📝 Demo 2: Batch Evaluation")
    print("-" * 50)
    
    from src.evaluator import RAGEvaluator
    from src.embeddings import EmbeddingService
    from src.chunker import Chunk
    
    # Initialize
    embedding_service = EmbeddingService()
    evaluator = RAGEvaluator(
        embedding_service=embedding_service,
        use_llm_evaluation=False,
    )
    
    # Sample chunks
    chunks = [
        Chunk(text="The bootcamp runs for 3 weeks.", chunk_id="c1", 
              source="bootcamp.docx", chunk_index=0),
        Chunk(text="Phase 1 covers document processing and chunking.", chunk_id="c2",
              source="bootcamp.docx", chunk_index=1),
    ]
    
    # Test cases
    test_cases = [
        {
            "question": "How long is the bootcamp?",
            "answer": "The bootcamp runs for 3 weeks.",
            "chunks": chunks,
            "ground_truth": "The bootcamp is a 3-week program.",
        },
        {
            "question": "What does Phase 1 cover?",
            "answer": "Phase 1 covers document processing and chunking strategies.",
            "chunks": chunks,
            "ground_truth": "Phase 1 focuses on document processing.",
        },
        {
            "question": "What is the cost?",
            "answer": "The program costs $5000.",  # Hallucinated
            "chunks": chunks,
            "ground_truth": None,
        },
        {
            "question": "Who teaches the bootcamp?",
            "answer": "I don't have information about the instructors.",
            "chunks": chunks,
            "ground_truth": None,
        },
    ]
    
    print(f"\nEvaluating {len(test_cases)} test cases...")
    
    batch_result = evaluator.evaluate_batch(test_cases)
    
    print("\n" + batch_result.summary())
    
    print("\n📋 Individual Results:")
    for i, result in enumerate(batch_result.results, 1):
        status = "✅" if result.passed else "❌"
        print(f"   {status} Case {i}: Score={result.overall_score:.2f}, "
              f"Faith={result.generation_metrics.faithfulness:.2f}, "
              f"Level={result.level.value}")


def demo_with_rag_agent():
    """Demonstrate evaluation with real RAG Agent."""
    print("\n" + "-" * 50)
    print("📝 Demo 3: Evaluation with RAG Agent")
    print("-" * 50)
    
    from src.evaluator import RAGEvaluator
    from src.embeddings import EmbeddingService
    from src.rag_agent import RAGAgent
    from src.chunker import Chunk
    import time
    
    print("\nInitializing RAG Agent...")
    
    try:
        agent = RAGAgent()
        
        # Ingest some sample data if needed
        data_dir = project_root / "data" / "knowledge_base"
        if data_dir.exists():
            print(f"Ingesting documents from {data_dir}...")
            agent.ingest_directory(str(data_dir))
    except Exception as e:
        print(f"⚠️  Could not initialize RAG Agent: {e}")
        print("   Skipping RAG Agent demo...")
        return
    
    # Initialize evaluator (access private attributes)
    evaluator = RAGEvaluator(
        embedding_service=agent._rag_chain.embedding_service,
        llm_service=agent._rag_chain.llm_service,
        use_llm_evaluation=True,  # Use LLM for faithfulness
    )
    
    # Test questions
    test_questions = [
        "What is the AI Bootcamp schedule?",
        "What happens in Week 1?",
        "How do I contact support?",
        "What programming languages are used?",
        "What is the deadline for the project?",
    ]
    
    print(f"\nEvaluating {len(test_questions)} questions with live RAG...\n")
    
    results = []
    for question in test_questions:
        print(f"🔍 Question: {question}")
        
        start_time = time.time()
        response = agent.query(question)
        latency = (time.time() - start_time) * 1000
        
        # Get chunks from response
        chunks = response.get("retrieved_chunks", [])
        if chunks and isinstance(chunks[0], str):
            # Convert to Chunk objects if needed
            chunks = [
                Chunk(text=text, chunk_id=f"c{i}", source="retrieved", chunk_index=i)
                for i, text in enumerate(chunks)
            ]
        
        # Evaluate
        result = evaluator.evaluate(
            question=question,
            answer=response.get("answer", ""),
            retrieved_chunks=chunks,
            latency_ms=latency,
        )
        results.append(result)
        
        # Print result
        status = "✅" if result.passed else "❌"
        print(f"   {status} Score: {result.overall_score:.2f} | "
              f"Faith: {result.generation_metrics.faithfulness:.2f} | "
              f"Rel: {result.generation_metrics.answer_relevance:.2f} | "
              f"Latency: {latency:.0f}ms")
        print(f"   Answer: {response.get('answer', '')[:80]}...")
        print()
    
    # Summary
    from src.evaluator import BatchEvaluationResult
    batch = BatchEvaluationResult(results=results)
    
    print("\n" + "=" * 50)
    print(batch.summary())
    print("=" * 50)


def demo_ground_truth_comparison():
    """Demonstrate evaluation against ground truth."""
    print("\n" + "-" * 50)
    print("📝 Demo 4: Ground Truth Comparison")
    print("-" * 50)
    
    from src.evaluator import RAGEvaluator
    from src.embeddings import EmbeddingService
    from src.chunker import Chunk
    
    # Initialize
    embedding_service = EmbeddingService()
    evaluator = RAGEvaluator(
        embedding_service=embedding_service,
        use_llm_evaluation=False,
    )
    
    # Sample chunks
    chunks = [
        Chunk(text="The AI Bootcamp is a 3-week program.", chunk_id="c1",
              source="bootcamp.docx", chunk_index=0),
    ]
    
    # Test with ground truth
    test_cases = [
        {
            "question": "How long is the bootcamp?",
            "answer": "The AI Bootcamp runs for 3 weeks.",
            "ground_truth": "The bootcamp is a 3-week intensive program.",
        },
        {
            "question": "How long is the bootcamp?",
            "answer": "The bootcamp is approximately one month long.",
            "ground_truth": "The bootcamp is a 3-week intensive program.",
        },
        {
            "question": "How long is the bootcamp?",
            "answer": "I don't have that information.",
            "ground_truth": "The bootcamp is a 3-week intensive program.",
        },
    ]
    
    print("\nComparing answers to ground truth:\n")
    
    for i, case in enumerate(test_cases, 1):
        result = evaluator.evaluate(
            question=case["question"],
            answer=case["answer"],
            retrieved_chunks=chunks,
            ground_truth=case["ground_truth"],
        )
        
        print(f"Case {i}:")
        print(f"   Answer: {case['answer']}")
        print(f"   Ground Truth: {case['ground_truth']}")
        print(f"   Similarity: {result.ground_truth_similarity:.2f}")
        print(f"   Overall Score: {result.overall_score:.2f}")
        print()


def main():
    print_banner()
    
    print("This demo shows how to evaluate RAG response quality.\n")
    print("Metrics evaluated:")
    print("  • Faithfulness - Is the answer grounded in context?")
    print("  • Answer Relevance - Is the answer relevant to the question?")
    print("  • Context Relevance - How relevant is the retrieved context?")
    print("  • Completeness - Does the answer fully address the question?")
    print("  • Coherence - Is the answer well-structured?")
    
    # Run demos
    demo_single_evaluation()
    demo_batch_evaluation()
    demo_ground_truth_comparison()
    
    # Optional: Run with real RAG agent
    print("\n" + "-" * 50)
    response = input("Run demo with live RAG Agent? [y/N]: ").strip().lower()
    if response == 'y':
        demo_with_rag_agent()
    
    print("\n✅ Evaluation demo complete!")
    print("\nTo use in your code:")
    print("""
    from src.evaluator import RAGEvaluator, create_evaluator
    
    evaluator = create_evaluator(use_llm=True)
    
    result = evaluator.evaluate(
        question="Your question",
        answer="The answer",
        retrieved_chunks=chunks,
    )
    
    print(f"Score: {result.overall_score}")
    print(f"Passed: {result.passed}")
    """)


if __name__ == "__main__":
    main()
