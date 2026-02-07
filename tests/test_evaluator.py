"""
Tests for RAG Evaluator Module

Tests for the evaluation metrics and scoring functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluator import (
    RAGEvaluator,
    RetrievalMetrics,
    GenerationMetrics,
    EvaluationResult,
    BatchEvaluationResult,
    MetricLevel,
    create_evaluator,
)
from src.chunker import Chunk
from src.vector_store import SearchResult


class TestMetricLevel:
    """Tests for MetricLevel enum."""
    
    def test_metric_levels_exist(self):
        assert MetricLevel.EXCELLENT.value == "excellent"
        assert MetricLevel.GOOD.value == "good"
        assert MetricLevel.FAIR.value == "fair"
        assert MetricLevel.POOR.value == "poor"


class TestRetrievalMetrics:
    """Tests for RetrievalMetrics dataclass."""
    
    def test_default_values(self):
        metrics = RetrievalMetrics()
        assert metrics.precision_at_k == 0.0
        assert metrics.recall_at_k == 0.0
        assert metrics.mrr == 0.0
        assert metrics.avg_similarity == 0.0
        assert metrics.context_relevance == 0.0
    
    def test_level_excellent(self):
        metrics = RetrievalMetrics(precision_at_k=0.9, context_relevance=0.85)
        assert metrics.level == MetricLevel.EXCELLENT
    
    def test_level_good(self):
        metrics = RetrievalMetrics(precision_at_k=0.7, context_relevance=0.65)
        assert metrics.level == MetricLevel.GOOD
    
    def test_level_fair(self):
        metrics = RetrievalMetrics(precision_at_k=0.5, context_relevance=0.45)
        assert metrics.level == MetricLevel.FAIR
    
    def test_level_poor(self):
        metrics = RetrievalMetrics(precision_at_k=0.2, context_relevance=0.1)
        assert metrics.level == MetricLevel.POOR
    
    def test_to_dict(self):
        metrics = RetrievalMetrics(
            precision_at_k=0.8,
            recall_at_k=0.7,
            mrr=0.5,
            avg_similarity=0.6,
            context_relevance=0.75,
        )
        d = metrics.to_dict()
        
        assert d["precision_at_k"] == 0.8
        assert d["recall_at_k"] == 0.7
        assert d["mrr"] == 0.5
        assert d["avg_similarity"] == 0.6
        assert d["context_relevance"] == 0.75
        assert "level" in d


class TestGenerationMetrics:
    """Tests for GenerationMetrics dataclass."""
    
    def test_default_values(self):
        metrics = GenerationMetrics()
        assert metrics.faithfulness == 0.0
        assert metrics.answer_relevance == 0.0
        assert metrics.completeness == 0.0
        assert metrics.coherence == 0.0
    
    def test_level_excellent(self):
        metrics = GenerationMetrics(faithfulness=0.9, answer_relevance=0.85)
        assert metrics.level == MetricLevel.EXCELLENT
    
    def test_level_good(self):
        metrics = GenerationMetrics(faithfulness=0.7, answer_relevance=0.65)
        assert metrics.level == MetricLevel.GOOD
    
    def test_level_fair(self):
        metrics = GenerationMetrics(faithfulness=0.5, answer_relevance=0.4)
        assert metrics.level == MetricLevel.FAIR
    
    def test_level_poor(self):
        metrics = GenerationMetrics(faithfulness=0.2, answer_relevance=0.1)
        assert metrics.level == MetricLevel.POOR
    
    def test_to_dict(self):
        metrics = GenerationMetrics(
            faithfulness=0.9,
            answer_relevance=0.85,
            completeness=0.8,
            coherence=0.7,
        )
        d = metrics.to_dict()
        
        assert d["faithfulness"] == 0.9
        assert d["answer_relevance"] == 0.85
        assert d["completeness"] == 0.8
        assert d["coherence"] == 0.7
        assert "level" in d


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""
    
    @pytest.fixture
    def sample_result(self):
        return EvaluationResult(
            question="What is the bootcamp schedule?",
            answer="The bootcamp runs for 3 weeks with hands-on projects.",
            retrieval_metrics=RetrievalMetrics(
                precision_at_k=0.8,
                context_relevance=0.75,
            ),
            generation_metrics=GenerationMetrics(
                faithfulness=0.9,
                answer_relevance=0.85,
                completeness=0.8,
                coherence=0.7,
            ),
            overall_score=0.82,
            latency_ms=150.0,
            num_chunks_retrieved=5,
            sources=["bootcamp.docx", "faq.docx"],
        )
    
    def test_level_from_score(self, sample_result):
        assert sample_result.level == MetricLevel.EXCELLENT
    
    def test_passed_with_good_metrics(self, sample_result):
        assert sample_result.passed is True
    
    def test_passed_fails_low_faithfulness(self):
        result = EvaluationResult(
            question="Test",
            answer="Test answer",
            retrieval_metrics=RetrievalMetrics(),
            generation_metrics=GenerationMetrics(faithfulness=0.3),
            overall_score=0.6,
        )
        assert result.passed is False
    
    def test_passed_fails_low_answer_relevance(self):
        result = EvaluationResult(
            question="Test",
            answer="Test answer",
            retrieval_metrics=RetrievalMetrics(),
            generation_metrics=GenerationMetrics(
                faithfulness=0.7,
                answer_relevance=0.2,
            ),
            overall_score=0.6,
        )
        assert result.passed is False
    
    def test_to_dict(self, sample_result):
        d = sample_result.to_dict()
        
        assert d["question"] == "What is the bootcamp schedule?"
        assert "answer" in d
        assert "retrieval" in d
        assert "generation" in d
        assert d["overall_score"] == 0.82
        assert d["passed"] is True
        assert d["num_chunks"] == 5
        assert len(d["sources"]) == 2
    
    def test_str_representation(self, sample_result):
        s = str(sample_result)
        assert "EvaluationResult" in s
        assert "overall_score" in s
        assert "passed" in s


class TestBatchEvaluationResult:
    """Tests for BatchEvaluationResult dataclass."""
    
    @pytest.fixture
    def sample_results(self):
        return [
            EvaluationResult(
                question="Q1",
                answer="A1",
                retrieval_metrics=RetrievalMetrics(),
                generation_metrics=GenerationMetrics(
                    faithfulness=0.9,
                    answer_relevance=0.8,
                ),
                overall_score=0.85,
                latency_ms=100,
            ),
            EvaluationResult(
                question="Q2",
                answer="A2",
                retrieval_metrics=RetrievalMetrics(),
                generation_metrics=GenerationMetrics(
                    faithfulness=0.7,
                    answer_relevance=0.6,
                ),
                overall_score=0.65,
                latency_ms=150,
            ),
            EvaluationResult(
                question="Q3",
                answer="A3",
                retrieval_metrics=RetrievalMetrics(),
                generation_metrics=GenerationMetrics(
                    faithfulness=0.4,
                    answer_relevance=0.3,
                ),
                overall_score=0.35,
                latency_ms=200,
            ),
        ]
    
    def test_avg_overall_score(self, sample_results):
        batch = BatchEvaluationResult(results=sample_results)
        expected = (0.85 + 0.65 + 0.35) / 3
        assert abs(batch.avg_overall_score - expected) < 0.01
    
    def test_avg_faithfulness(self, sample_results):
        batch = BatchEvaluationResult(results=sample_results)
        expected = (0.9 + 0.7 + 0.4) / 3
        assert abs(batch.avg_faithfulness - expected) < 0.01
    
    def test_avg_answer_relevance(self, sample_results):
        batch = BatchEvaluationResult(results=sample_results)
        expected = (0.8 + 0.6 + 0.3) / 3
        assert abs(batch.avg_answer_relevance - expected) < 0.01
    
    def test_pass_rate(self, sample_results):
        batch = BatchEvaluationResult(results=sample_results)
        # First 2 pass (faithfulness >= 0.5, answer_relevance >= 0.4, overall >= 0.5)
        # Third fails
        assert batch.pass_rate == 2/3
    
    def test_avg_latency(self, sample_results):
        batch = BatchEvaluationResult(results=sample_results)
        expected = (100 + 150 + 200) / 3
        assert abs(batch.avg_latency_ms - expected) < 0.01
    
    def test_empty_results(self):
        batch = BatchEvaluationResult(results=[])
        assert batch.avg_overall_score == 0.0
        assert batch.pass_rate == 0.0
        assert batch.avg_latency_ms == 0.0
    
    def test_to_dict(self, sample_results):
        batch = BatchEvaluationResult(results=sample_results)
        d = batch.to_dict()
        
        assert d["num_evaluated"] == 3
        assert "avg_overall_score" in d
        assert "avg_faithfulness" in d
        assert "pass_rate" in d
        assert len(d["results"]) == 3
    
    def test_summary(self, sample_results):
        batch = BatchEvaluationResult(results=sample_results)
        summary = batch.summary()
        
        assert "Batch Evaluation Summary" in summary
        assert "Total: 3" in summary
        assert "Pass Rate" in summary


class TestRAGEvaluator:
    """Tests for RAGEvaluator class."""
    
    @pytest.fixture
    def mock_embedding_service(self):
        service = Mock()
        # Return consistent embeddings for testing
        service.embed_query.side_effect = lambda text: [0.1] * 384
        return service
    
    @pytest.fixture
    def mock_llm_service(self):
        service = Mock()
        service.generate.return_value = "0.85"
        return service
    
    @pytest.fixture
    def sample_chunks(self):
        return [
            Chunk(
                text="The AI Bootcamp runs for 3 weeks.",
                chunk_id="c1",
                source="bootcamp.docx",
                chunk_index=0,
            ),
            Chunk(
                text="Week 1 focuses on RAG fundamentals.",
                chunk_id="c2",
                source="bootcamp.docx",
                chunk_index=1,
            ),
        ]
    
    def test_initialization_without_llm(self, mock_embedding_service):
        evaluator = RAGEvaluator(
            embedding_service=mock_embedding_service,
            llm_service=None,
            use_llm_evaluation=False,
        )
        assert evaluator.use_llm_evaluation is False
    
    def test_initialization_with_llm(self, mock_embedding_service, mock_llm_service):
        evaluator = RAGEvaluator(
            embedding_service=mock_embedding_service,
            llm_service=mock_llm_service,
            use_llm_evaluation=True,
        )
        assert evaluator.use_llm_evaluation is True
    
    def test_evaluate_basic(self, mock_embedding_service, sample_chunks):
        evaluator = RAGEvaluator(
            embedding_service=mock_embedding_service,
            use_llm_evaluation=False,
        )
        
        result = evaluator.evaluate(
            question="What is the bootcamp schedule?",
            answer="The bootcamp runs for 3 weeks with hands-on projects.",
            retrieved_chunks=sample_chunks,
        )
        
        assert isinstance(result, EvaluationResult)
        assert result.question == "What is the bootcamp schedule?"
        assert result.num_chunks_retrieved == 2
        assert "bootcamp.docx" in result.sources
    
    def test_evaluate_with_search_results(self, mock_embedding_service, sample_chunks):
        evaluator = RAGEvaluator(
            embedding_service=mock_embedding_service,
            use_llm_evaluation=False,
        )
        
        search_results = [
            SearchResult(chunk=sample_chunks[0], score=0.9, rank=1),
            SearchResult(chunk=sample_chunks[1], score=0.8, rank=2),
        ]
        
        result = evaluator.evaluate(
            question="What is the schedule?",
            answer="The schedule covers 3 weeks.",
            search_results=search_results,
        )
        
        # Use pytest.approx for floating-point comparison
        assert result.retrieval_metrics.avg_similarity == pytest.approx(0.85, rel=1e-9)
    
    def test_evaluate_with_ground_truth(self, mock_embedding_service, sample_chunks):
        evaluator = RAGEvaluator(
            embedding_service=mock_embedding_service,
            use_llm_evaluation=False,
        )
        
        result = evaluator.evaluate(
            question="What is the schedule?",
            answer="The bootcamp is 3 weeks long.",
            retrieved_chunks=sample_chunks,
            ground_truth="The AI Bootcamp runs for 3 weeks.",
        )
        
        assert result.ground_truth_similarity is not None
        assert result.ground_truth_similarity >= 0.0
    
    def test_evaluate_empty_chunks(self, mock_embedding_service):
        evaluator = RAGEvaluator(
            embedding_service=mock_embedding_service,
            use_llm_evaluation=False,
        )
        
        result = evaluator.evaluate(
            question="What is the schedule?",
            answer="I don't have information about that.",
            retrieved_chunks=[],
        )
        
        assert result.num_chunks_retrieved == 0
        assert result.retrieval_metrics.context_relevance == 0.0
    
    def test_evaluate_empty_answer(self, mock_embedding_service, sample_chunks):
        evaluator = RAGEvaluator(
            embedding_service=mock_embedding_service,
            use_llm_evaluation=False,
        )
        
        result = evaluator.evaluate(
            question="What is the schedule?",
            answer="",
            retrieved_chunks=sample_chunks,
        )
        
        assert result.generation_metrics.faithfulness == 0.0
        assert result.generation_metrics.answer_relevance == 0.0
    
    def test_evaluate_with_llm(self, mock_embedding_service, mock_llm_service, sample_chunks):
        evaluator = RAGEvaluator(
            embedding_service=mock_embedding_service,
            llm_service=mock_llm_service,
            use_llm_evaluation=True,
        )
        
        result = evaluator.evaluate(
            question="What is the schedule?",
            answer="The bootcamp is 3 weeks.",
            retrieved_chunks=sample_chunks,
        )
        
        # LLM should be called for faithfulness/completeness
        assert mock_llm_service.generate.called
    
    def test_evaluate_batch(self, mock_embedding_service, sample_chunks):
        evaluator = RAGEvaluator(
            embedding_service=mock_embedding_service,
            use_llm_evaluation=False,
        )
        
        test_cases = [
            {
                "question": "Q1",
                "answer": "A1",
                "chunks": sample_chunks,
            },
            {
                "question": "Q2",
                "answer": "A2",
                "chunks": sample_chunks,
            },
        ]
        
        batch_result = evaluator.evaluate_batch(test_cases)
        
        assert isinstance(batch_result, BatchEvaluationResult)
        assert len(batch_result.results) == 2
    
    def test_estimate_coherence_short_answer(self, mock_embedding_service):
        evaluator = RAGEvaluator(
            embedding_service=mock_embedding_service,
            use_llm_evaluation=False,
        )
        
        # Short answer should have lower coherence
        short_score = evaluator._estimate_coherence("Yes")
        normal_score = evaluator._estimate_coherence("The bootcamp runs for 3 weeks with hands-on projects.")
        
        assert short_score < normal_score
    
    def test_estimate_coherence_structured_answer(self, mock_embedding_service):
        evaluator = RAGEvaluator(
            embedding_service=mock_embedding_service,
            use_llm_evaluation=False,
        )
        
        # Structured answer should have higher coherence
        structured = "The bootcamp has: 1. Week 1 - Basics. 2. Week 2 - Advanced."
        unstructured = "bootcamp week basics advanced"
        
        structured_score = evaluator._estimate_coherence(structured)
        unstructured_score = evaluator._estimate_coherence(unstructured)
        
        assert structured_score > unstructured_score
    
    def test_estimate_faithfulness_high_overlap(self, mock_embedding_service):
        evaluator = RAGEvaluator(
            embedding_service=mock_embedding_service,
            use_llm_evaluation=False,
        )
        
        context = "The AI Bootcamp runs for 3 weeks with hands-on projects."
        answer = "The bootcamp runs for 3 weeks."
        
        score = evaluator._estimate_faithfulness(answer, context)
        assert score > 0.5
    
    def test_estimate_faithfulness_low_overlap(self, mock_embedding_service):
        evaluator = RAGEvaluator(
            embedding_service=mock_embedding_service,
            use_llm_evaluation=False,
        )
        
        context = "The AI Bootcamp runs for 3 weeks."
        answer = "Machine learning uses neural networks for prediction."
        
        score = evaluator._estimate_faithfulness(answer, context)
        assert score < 0.5
    
    def test_parse_score_valid(self, mock_embedding_service):
        evaluator = RAGEvaluator(
            embedding_service=mock_embedding_service,
            use_llm_evaluation=False,
        )
        
        assert evaluator._parse_score("0.85") == 0.85
        assert evaluator._parse_score("0.5") == 0.5
        assert evaluator._parse_score("Score: 0.7") == 0.7
    
    def test_parse_score_normalized(self, mock_embedding_service):
        evaluator = RAGEvaluator(
            embedding_service=mock_embedding_service,
            use_llm_evaluation=False,
        )
        
        # Score > 1 should be normalized
        assert evaluator._parse_score("8.5") == 0.85
        assert evaluator._parse_score("85") == 0.85
    
    def test_parse_score_invalid(self, mock_embedding_service):
        evaluator = RAGEvaluator(
            embedding_service=mock_embedding_service,
            use_llm_evaluation=False,
        )
        
        # Invalid should return default
        assert evaluator._parse_score("invalid") == 0.5
        assert evaluator._parse_score("") == 0.5


class TestCreateEvaluator:
    """Tests for create_evaluator factory function."""
    
    def test_create_without_services(self):
        with patch('src.evaluator.EmbeddingService') as MockEmbedding:
            mock_embedding = Mock()
            MockEmbedding.return_value = mock_embedding
            
            evaluator = create_evaluator()
            
            assert isinstance(evaluator, RAGEvaluator)
    
    def test_create_with_services(self):
        mock_embedding = Mock()
        mock_llm = Mock()
        
        evaluator = create_evaluator(
            embedding_service=mock_embedding,
            llm_service=mock_llm,
            use_llm=True,
        )
        
        assert evaluator.embedding_service == mock_embedding
        assert evaluator.llm_service == mock_llm
        assert evaluator.use_llm_evaluation is True
    
    def test_create_without_llm_evaluation(self):
        mock_embedding = Mock()
        
        evaluator = create_evaluator(
            embedding_service=mock_embedding,
            use_llm=False,
        )
        
        assert evaluator.use_llm_evaluation is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
