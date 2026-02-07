"""Quick demo of RAG Evaluator."""
from src.evaluator import create_evaluator

# Create evaluator (sans LLM pour démo rapide)
evaluator = create_evaluator(use_llm_evaluation=False)
print("RAG Evaluator créé avec succès!")
print()

# Simulate a RAG evaluation
chunks = [
    "L'AI Bootcamp dure 3 semaines.",
    "Les sessions sont le lundi et mercredi.",
    "Formation pratique avec projets réels."
]

result = evaluator.evaluate(
    question="Quelle est la durée du bootcamp?",
    answer="Le bootcamp AI dure 3 semaines avec des sessions le lundi et mercredi.",
    retrieved_chunks=chunks,
)

print("=== Résultats Évaluation RAG ===")
print()
print(f"Question: {result.question}")
print(f"Réponse: {result.answer}")
print()
print("--- Métriques de Retrieval ---")
print(f"  Precision@K: {result.retrieval_metrics.precision_at_k:.2f}")
print(f"  Context Relevance: {result.retrieval_metrics.context_relevance:.2f}")
print(f"  Niveau: {result.retrieval_metrics.level.value}")
print()
print("--- Métriques de Génération ---")
print(f"  Faithfulness: {result.generation_metrics.faithfulness:.2f}")
print(f"  Answer Relevance: {result.generation_metrics.answer_relevance:.2f}")
print(f"  Completeness: {result.generation_metrics.completeness:.2f}")
print(f"  Coherence: {result.generation_metrics.coherence:.2f}")
print(f"  Niveau: {result.generation_metrics.level.value}")
print()
print(f"SCORE GLOBAL: {result.overall_score:.2f}")
print(f"STATUT: {'PASS ✓' if result.passed else 'FAIL ✗'}")
