"""
RAG Evaluation Module

Provides metrics to evaluate the quality of RAG (Retrieval-Augmented Generation) responses.

Metrics implemented:
1. Retrieval Metrics:
   - Precision@K: How many retrieved chunks are relevant
   - Recall@K: How many relevant chunks were retrieved
   - MRR (Mean Reciprocal Rank): Position of first relevant result

2. Generation Metrics:
   - Faithfulness: Is the answer grounded in retrieved context? (no hallucination)
   - Answer Relevance: How relevant is the answer to the question?
   - Context Relevance: How relevant is the retrieved context to the question?

3. End-to-End Metrics:
   - Response Quality Score: Combined quality metric
   - Latency tracking: Response time measurement

Design Rationale:
- Uses LLM-as-judge for faithfulness evaluation (cost-effective)
- Embedding similarity for relevance metrics
- Configurable thresholds for different use cases
- Supports both automated and human evaluation

Usage:
    evaluator = RAGEvaluator(embedding_service, llm_service)
    
    # Evaluate a single response
    result = evaluator.evaluate(
        question="What is the bootcamp schedule?",
        answer="The bootcamp runs for 3 weeks...",
        retrieved_chunks=chunks,
        ground_truth="The bootcamp is a 3-week program..."  # optional
    )
    
    # Run batch evaluation
    results = evaluator.evaluate_batch(test_cases)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import numpy as np

from src.embeddings import EmbeddingService, cosine_similarity
from src.llm_service import LLMService
from src.chunker import Chunk
from src.vector_store import SearchResult

logger = logging.getLogger(__name__)


class MetricLevel(Enum):
    """Quality level based on metric scores."""
    EXCELLENT = "excellent"  # >= 0.8
    GOOD = "good"           # >= 0.6
    FAIR = "fair"           # >= 0.4
    POOR = "poor"           # < 0.4


@dataclass
class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""
    
    precision_at_k: float = 0.0  # Fraction of retrieved docs that are relevant
    recall_at_k: float = 0.0     # Fraction of relevant docs that were retrieved
    mrr: float = 0.0             # Mean Reciprocal Rank
    avg_similarity: float = 0.0  # Average similarity score of retrieved chunks
    context_relevance: float = 0.0  # How relevant is context to query
    
    @property
    def level(self) -> MetricLevel:
        """Get quality level based on average metrics."""
        avg = (self.precision_at_k + self.context_relevance) / 2
        if avg >= 0.8:
            return MetricLevel.EXCELLENT
        elif avg >= 0.6:
            return MetricLevel.GOOD
        elif avg >= 0.4:
            return MetricLevel.FAIR
        return MetricLevel.POOR
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision_at_k": round(self.precision_at_k, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "avg_similarity": round(self.avg_similarity, 4),
            "context_relevance": round(self.context_relevance, 4),
            "level": self.level.value,
        }


@dataclass
class GenerationMetrics:
    """Metrics for evaluating generation quality."""
    
    faithfulness: float = 0.0      # Is answer grounded in context?
    answer_relevance: float = 0.0  # Is answer relevant to question?
    completeness: float = 0.0      # Does answer fully address the question?
    coherence: float = 0.0         # Is answer well-structured?
    
    @property
    def level(self) -> MetricLevel:
        """Get quality level based on average metrics."""
        avg = (self.faithfulness + self.answer_relevance) / 2
        if avg >= 0.8:
            return MetricLevel.EXCELLENT
        elif avg >= 0.6:
            return MetricLevel.GOOD
        elif avg >= 0.4:
            return MetricLevel.FAIR
        return MetricLevel.POOR
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "completeness": round(self.completeness, 4),
            "coherence": round(self.coherence, 4),
            "level": self.level.value,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result for a RAG response."""
    
    question: str
    answer: str
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    
    # Overall scores
    overall_score: float = 0.0
    latency_ms: float = 0.0
    
    # Metadata
    num_chunks_retrieved: int = 0
    sources: List[str] = field(default_factory=list)
    
    # Optional ground truth comparison
    ground_truth_similarity: Optional[float] = None
    
    @property
    def level(self) -> MetricLevel:
        """Get overall quality level."""
        if self.overall_score >= 0.8:
            return MetricLevel.EXCELLENT
        elif self.overall_score >= 0.6:
            return MetricLevel.GOOD
        elif self.overall_score >= 0.4:
            return MetricLevel.FAIR
        return MetricLevel.POOR
    
    @property
    def passed(self) -> bool:
        """Check if evaluation passed minimum thresholds."""
        return (
            self.generation_metrics.faithfulness >= 0.5 and
            self.generation_metrics.answer_relevance >= 0.4 and
            self.overall_score >= 0.5
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer[:200] + "..." if len(self.answer) > 200 else self.answer,
            "retrieval": self.retrieval_metrics.to_dict(),
            "generation": self.generation_metrics.to_dict(),
            "overall_score": round(self.overall_score, 4),
            "level": self.level.value,
            "passed": self.passed,
            "latency_ms": round(self.latency_ms, 2),
            "num_chunks": self.num_chunks_retrieved,
            "sources": self.sources,
            "ground_truth_similarity": (
                round(self.ground_truth_similarity, 4) 
                if self.ground_truth_similarity else None
            ),
        }
    
    def __str__(self) -> str:
        return (
            f"EvaluationResult(\n"
            f"  question='{self.question[:50]}...'\n"
            f"  overall_score={self.overall_score:.2f} ({self.level.value})\n"
            f"  faithfulness={self.generation_metrics.faithfulness:.2f}\n"
            f"  answer_relevance={self.generation_metrics.answer_relevance:.2f}\n"
            f"  context_relevance={self.retrieval_metrics.context_relevance:.2f}\n"
            f"  passed={self.passed}\n"
            f")"
        )


@dataclass
class BatchEvaluationResult:
    """Results from evaluating multiple test cases."""
    
    results: List[EvaluationResult]
    
    @property
    def avg_overall_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.overall_score for r in self.results) / len(self.results)
    
    @property
    def avg_faithfulness(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.generation_metrics.faithfulness for r in self.results) / len(self.results)
    
    @property
    def avg_answer_relevance(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.generation_metrics.answer_relevance for r in self.results) / len(self.results)
    
    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)
    
    @property
    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_evaluated": len(self.results),
            "avg_overall_score": round(self.avg_overall_score, 4),
            "avg_faithfulness": round(self.avg_faithfulness, 4),
            "avg_answer_relevance": round(self.avg_answer_relevance, 4),
            "pass_rate": round(self.pass_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "results": [r.to_dict() for r in self.results],
        }
    
    def summary(self) -> str:
        """Get a summary string."""
        return (
            f"Batch Evaluation Summary:\n"
            f"  Total: {len(self.results)} test cases\n"
            f"  Pass Rate: {self.pass_rate:.1%}\n"
            f"  Avg Score: {self.avg_overall_score:.2f}\n"
            f"  Avg Faithfulness: {self.avg_faithfulness:.2f}\n"
            f"  Avg Answer Relevance: {self.avg_answer_relevance:.2f}\n"
            f"  Avg Latency: {self.avg_latency_ms:.0f}ms"
        )


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality using multiple metrics.
    
    Provides both embedding-based and LLM-based evaluation methods.
    
    Example:
        evaluator = RAGEvaluator(embedding_service, llm_service)
        
        result = evaluator.evaluate(
            question="What is the schedule?",
            answer="The schedule is...",
            retrieved_chunks=chunks
        )
        
        print(f"Score: {result.overall_score}")
        print(f"Passed: {result.passed}")
    """
    
    # Prompts for LLM-based evaluation
    FAITHFULNESS_PROMPT = """You are evaluating whether an answer is faithful to the given context.

Context:
{context}

Question: {question}

Answer: {answer}

Evaluate if the answer is ONLY based on information from the context (no hallucination).

Scoring:
- 1.0: Answer is completely grounded in context
- 0.7: Answer is mostly grounded, minor extrapolation
- 0.5: Answer partially uses context but adds unsupported claims
- 0.3: Answer has significant unsupported information
- 0.0: Answer contradicts context or is completely made up

Respond with ONLY a single number between 0.0 and 1.0."""

    COMPLETENESS_PROMPT = """You are evaluating whether an answer completely addresses the question.

Question: {question}

Answer: {answer}

Context Available: {context_summary}

Scoring:
- 1.0: Answer fully addresses all aspects of the question
- 0.7: Answer addresses most aspects
- 0.5: Answer partially addresses the question
- 0.3: Answer barely addresses the question
- 0.0: Answer does not address the question at all

Respond with ONLY a single number between 0.0 and 1.0."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        llm_service: Optional[LLMService] = None,
        use_llm_evaluation: bool = True,
    ):
        """
        Initialize the RAG evaluator.
        
        Args:
            embedding_service: For similarity-based metrics
            llm_service: For LLM-as-judge metrics
            use_llm_evaluation: Whether to use LLM for faithfulness eval
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.llm_service = llm_service
        self.use_llm_evaluation = use_llm_evaluation and llm_service is not None
        
        logger.info(
            f"RAGEvaluator initialized: "
            f"llm_eval={self.use_llm_evaluation}"
        )
    
    def evaluate(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[Chunk] = None,
        search_results: List[SearchResult] = None,
        ground_truth: Optional[str] = None,
        relevant_chunk_ids: Optional[List[str]] = None,
        latency_ms: float = 0.0,
    ) -> EvaluationResult:
        """
        Evaluate a single RAG response.
        
        Args:
            question: The user's question
            answer: The generated answer
            retrieved_chunks: List of retrieved Chunk objects
            search_results: Alternative: SearchResult objects with scores
            ground_truth: Optional expected answer for comparison
            relevant_chunk_ids: Optional list of known-relevant chunk IDs
            latency_ms: Response latency in milliseconds
            
        Returns:
            EvaluationResult with all metrics
        """
        start_time = time.time()
        
        # Normalize inputs
        if search_results and not retrieved_chunks:
            retrieved_chunks = [r.chunk for r in search_results]
            similarity_scores = [r.score for r in search_results]
        elif retrieved_chunks:
            similarity_scores = []
        else:
            retrieved_chunks = []
            similarity_scores = []
        
        # Calculate retrieval metrics
        retrieval_metrics = self._evaluate_retrieval(
            question=question,
            chunks=retrieved_chunks,
            similarity_scores=similarity_scores,
            relevant_chunk_ids=relevant_chunk_ids,
        )
        
        # Calculate generation metrics
        generation_metrics = self._evaluate_generation(
            question=question,
            answer=answer,
            context="\n\n".join([c.text for c in retrieved_chunks]),
        )
        
        # Calculate ground truth similarity if provided
        gt_similarity = None
        if ground_truth:
            gt_similarity = self._calculate_similarity(answer, ground_truth)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            retrieval_metrics,
            generation_metrics,
            gt_similarity,
        )
        
        # Get sources
        sources = list(set(c.source for c in retrieved_chunks))
        
        eval_time = (time.time() - start_time) * 1000
        
        result = EvaluationResult(
            question=question,
            answer=answer,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            overall_score=overall_score,
            latency_ms=latency_ms or eval_time,
            num_chunks_retrieved=len(retrieved_chunks),
            sources=sources,
            ground_truth_similarity=gt_similarity,
        )
        
        logger.info(
            f"Evaluation complete: score={overall_score:.2f}, "
            f"level={result.level.value}, passed={result.passed}"
        )
        
        return result
    
    def _evaluate_retrieval(
        self,
        question: str,
        chunks: List[Chunk],
        similarity_scores: List[float],
        relevant_chunk_ids: Optional[List[str]] = None,
    ) -> RetrievalMetrics:
        """Evaluate retrieval quality."""
        metrics = RetrievalMetrics()
        
        if not chunks:
            return metrics
        
        # Average similarity score
        if similarity_scores:
            metrics.avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # Context relevance: how relevant is the combined context to the question
        context = " ".join([c.text for c in chunks])
        metrics.context_relevance = self._calculate_similarity(question, context)
        
        # If we have ground truth relevance labels
        if relevant_chunk_ids:
            retrieved_ids = set(c.chunk_id for c in chunks)
            relevant_set = set(relevant_chunk_ids)
            
            # Precision@K
            relevant_retrieved = retrieved_ids.intersection(relevant_set)
            metrics.precision_at_k = len(relevant_retrieved) / len(chunks)
            
            # Recall@K
            if relevant_set:
                metrics.recall_at_k = len(relevant_retrieved) / len(relevant_set)
            
            # MRR (Mean Reciprocal Rank)
            for i, chunk in enumerate(chunks, 1):
                if chunk.chunk_id in relevant_set:
                    metrics.mrr = 1.0 / i
                    break
        else:
            # Estimate precision based on similarity threshold
            # Chunks with similarity > 0.5 are considered "relevant"
            if similarity_scores:
                relevant_count = sum(1 for s in similarity_scores if s > 0.5)
                metrics.precision_at_k = relevant_count / len(similarity_scores)
            else:
                # Use context relevance as proxy
                metrics.precision_at_k = metrics.context_relevance
        
        return metrics
    
    def _evaluate_generation(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> GenerationMetrics:
        """Evaluate generation quality."""
        metrics = GenerationMetrics()
        
        if not answer or answer.strip() == "":
            return metrics
        
        # Answer relevance: similarity between question and answer
        metrics.answer_relevance = self._calculate_similarity(question, answer)
        
        # Coherence: estimate based on answer structure
        metrics.coherence = self._estimate_coherence(answer)
        
        # Faithfulness and completeness: use LLM if available
        if self.use_llm_evaluation and context:
            metrics.faithfulness = self._evaluate_faithfulness_llm(
                question, answer, context
            )
            metrics.completeness = self._evaluate_completeness_llm(
                question, answer, context
            )
        else:
            # Fallback: estimate faithfulness from context overlap
            metrics.faithfulness = self._estimate_faithfulness(answer, context)
            metrics.completeness = min(metrics.answer_relevance * 1.2, 1.0)
        
        return metrics
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        try:
            emb1 = self.embedding_service.embed_query(text1)
            emb2 = self.embedding_service.embed_query(text2)
            return cosine_similarity(emb1, emb2)
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _estimate_coherence(self, answer: str) -> float:
        """Estimate answer coherence based on structure."""
        score = 0.5  # Base score
        
        # Penalize very short answers
        if len(answer) < 20:
            score -= 0.2
        elif len(answer) > 50:
            score += 0.1
        
        # Reward proper sentences
        if answer.endswith(('.', '!', '?')):
            score += 0.1
        
        # Reward structured content
        if any(marker in answer for marker in ['1.', '2.', '-', '•', ':']):
            score += 0.1
        
        # Penalize "I don't know" type responses
        negative_phrases = ["i don't have", "i cannot", "no information", "not found"]
        if any(phrase in answer.lower() for phrase in negative_phrases):
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _estimate_faithfulness(self, answer: str, context: str) -> float:
        """Estimate faithfulness without LLM (word overlap based)."""
        if not context:
            return 0.0
        
        # Tokenize
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                      'from', 'as', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'between', 'under', 'again', 'further',
                      'then', 'once', 'here', 'there', 'when', 'where', 'why',
                      'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                      'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                      'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                      'because', 'until', 'while', 'this', 'that', 'these', 'those'}
        
        answer_content = answer_words - stop_words
        context_content = context_words - stop_words
        
        if not answer_content:
            return 0.5
        
        # Calculate overlap
        overlap = answer_content.intersection(context_content)
        faithfulness = len(overlap) / len(answer_content)
        
        return min(faithfulness, 1.0)
    
    def _evaluate_faithfulness_llm(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> float:
        """Evaluate faithfulness using LLM-as-judge."""
        if not self.llm_service:
            return self._estimate_faithfulness(answer, context)
        
        try:
            prompt = self.FAITHFULNESS_PROMPT.format(
                context=context[:2000],  # Limit context length
                question=question,
                answer=answer,
            )
            
            response = self.llm_service.generate(prompt)
            
            # Parse score from response
            score = self._parse_score(response)
            return score
            
        except Exception as e:
            logger.warning(f"LLM faithfulness evaluation failed: {e}")
            return self._estimate_faithfulness(answer, context)
    
    def _evaluate_completeness_llm(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> float:
        """Evaluate completeness using LLM-as-judge."""
        if not self.llm_service:
            return 0.5
        
        try:
            context_summary = context[:500] + "..." if len(context) > 500 else context
            
            prompt = self.COMPLETENESS_PROMPT.format(
                question=question,
                answer=answer,
                context_summary=context_summary,
            )
            
            response = self.llm_service.generate(prompt)
            return self._parse_score(response)
            
        except Exception as e:
            logger.warning(f"LLM completeness evaluation failed: {e}")
            return 0.5
    
    def _parse_score(self, response: str) -> float:
        """Parse a numeric score from LLM response."""
        try:
            # Clean response
            response = response.strip()
            
            # Try to extract number
            import re
            matches = re.findall(r'(\d+\.?\d*)', response)
            
            if matches:
                score = float(matches[0])
                # Normalize if needed
                if score > 1.0:
                    score = score / 10.0 if score <= 10 else score / 100.0
                return max(0.0, min(1.0, score))
            
            return 0.5  # Default
            
        except Exception:
            return 0.5
    
    def _calculate_overall_score(
        self,
        retrieval: RetrievalMetrics,
        generation: GenerationMetrics,
        gt_similarity: Optional[float],
    ) -> float:
        """Calculate weighted overall score."""
        # Weights for different metrics
        weights = {
            'faithfulness': 0.30,      # Most important - no hallucination
            'answer_relevance': 0.25,  # Answer should be relevant
            'context_relevance': 0.20, # Retrieved context quality
            'completeness': 0.15,      # Answer completeness
            'coherence': 0.10,         # Answer quality
        }
        
        score = (
            weights['faithfulness'] * generation.faithfulness +
            weights['answer_relevance'] * generation.answer_relevance +
            weights['context_relevance'] * retrieval.context_relevance +
            weights['completeness'] * generation.completeness +
            weights['coherence'] * generation.coherence
        )
        
        # Bonus for matching ground truth
        if gt_similarity and gt_similarity > 0.7:
            score = min(1.0, score + 0.1)
        
        return score
    
    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> BatchEvaluationResult:
        """
        Evaluate multiple test cases.
        
        Args:
            test_cases: List of dicts with keys:
                - question: str
                - answer: str
                - chunks: List[Chunk] (optional)
                - ground_truth: str (optional)
                - latency_ms: float (optional)
                
        Returns:
            BatchEvaluationResult with aggregate metrics
        """
        results = []
        
        for i, case in enumerate(test_cases):
            logger.info(f"Evaluating test case {i+1}/{len(test_cases)}")
            
            result = self.evaluate(
                question=case.get('question', ''),
                answer=case.get('answer', ''),
                retrieved_chunks=case.get('chunks', []),
                search_results=case.get('search_results', []),
                ground_truth=case.get('ground_truth'),
                latency_ms=case.get('latency_ms', 0),
            )
            results.append(result)
        
        return BatchEvaluationResult(results=results)


def create_evaluator(
    embedding_service: Optional[EmbeddingService] = None,
    llm_service: Optional[LLMService] = None,
    use_llm: bool = True,
) -> RAGEvaluator:
    """
    Factory function to create a RAG evaluator.
    
    Args:
        embedding_service: Optional embedding service
        llm_service: Optional LLM service for advanced evaluation
        use_llm: Whether to use LLM-based evaluation
        
    Returns:
        Configured RAGEvaluator instance
    """
    return RAGEvaluator(
        embedding_service=embedding_service,
        llm_service=llm_service,
        use_llm_evaluation=use_llm,
    )
