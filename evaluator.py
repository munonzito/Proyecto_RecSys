import numpy as np
from typing import List, Dict
from difflib import SequenceMatcher

class RecommendationEvaluator:
    """Calcula métricas de evaluación para recomendaciones"""
    
    def __init__(self, fuzzy_match_threshold: float = 0.8):
        """
        Args:
            fuzzy_match_threshold: umbral de similitud para considerar match (0-1)
        """
        self.fuzzy_match_threshold = fuzzy_match_threshold
    
    def normalize_title(self, title: str) -> str:
        """Normaliza un título de película para comparación"""
        # Minúsculas
        title = title.lower()
        # Remover año si está presente
        import re
        title = re.sub(r'\s*\(\d{4}\)\s*', '', title)
        # Remover espacios extras
        title = ' '.join(title.split())
        # Remover puntuación común
        title = title.replace(',', '').replace('.', '').replace(':', '').replace('the ', '')
        return title.strip()
    
    def fuzzy_match(self, title1: str, title2: str) -> float:
        """Calcula similitud entre dos títulos (0-1)"""
        norm1 = self.normalize_title(title1)
        norm2 = self.normalize_title(title2)
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def match_titles(self, recommended: List[str], ground_truth: List[str]) -> List[str]:
        """
        Encuentra matches entre recomendaciones y ground truth usando fuzzy matching
        
        Returns:
            Lista de títulos de ground_truth que fueron recomendados
        """
        matched = []
        
        for gt_title in ground_truth:
            for rec_title in recommended:
                similarity = self.fuzzy_match(gt_title, rec_title)
                if similarity >= self.fuzzy_match_threshold:
                    matched.append(gt_title)
                    break  # ya encontramos match para este gt_title
        
        return matched
    
    def recall_at_k(self, recommended: List[str], ground_truth: List[str], k: int = 10) -> float:
        """
        Recall@K: proporción de items relevantes que fueron recomendados
        """
        if len(ground_truth) == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        matched = self.match_titles(recommended_k, ground_truth)
        
        return len(matched) / len(ground_truth)
    
    def precision_at_k(self, recommended: List[str], ground_truth: List[str], k: int = 10) -> float:
        """
        Precision@K: proporción de items recomendados que son relevantes
        """
        if len(recommended) == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        matched = self.match_titles(recommended_k, ground_truth)
        
        return len(matched) / min(len(recommended_k), k)
    
    def ndcg_at_k(self, recommended: List[str], ground_truth: List[str], k: int = 10) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        """
        recommended_k = recommended[:k]
        
        # DCG
        dcg = 0.0
        for i, rec_title in enumerate(recommended_k):
            # Verificar si este título hace match con alguno del ground truth
            for gt_title in ground_truth:
                if self.fuzzy_match(rec_title, gt_title) >= self.fuzzy_match_threshold:
                    dcg += 1.0 / np.log2(i + 2)  # +2 porque empieza en 0
                    break
        
        # IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(len(ground_truth), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def hit_rate_at_k(self, recommended: List[str], ground_truth: List[str], k: int = 10) -> float:
        """
        Hit Rate@K: 1 si al menos 1 item relevante está en top-K, 0 si no
        """
        recommended_k = recommended[:k]
        matched = self.match_titles(recommended_k, ground_truth)
        return 1.0 if len(matched) > 0 else 0.0
    
    def mrr(self, recommended: List[str], ground_truth: List[str]) -> float:
        """
        Mean Reciprocal Rank: 1/rank del primer item relevante
        """
        for i, rec_title in enumerate(recommended):
            for gt_title in ground_truth:
                if self.fuzzy_match(rec_title, gt_title) >= self.fuzzy_match_threshold:
                    return 1.0 / (i + 1)
        return 0.0
    
    def evaluate_all(self, recommended: List[str], ground_truth: List[str], 
                     k_values: List[int] = [5, 10]) -> Dict[str, float]:
        """
        Calcula todas las métricas
        
        Returns:
            dict con todas las métricas
        """
        metrics = {}
        
        for k in k_values:
            metrics[f'recall@{k}'] = self.recall_at_k(recommended, ground_truth, k)
            metrics[f'precision@{k}'] = self.precision_at_k(recommended, ground_truth, k)
            metrics[f'ndcg@{k}'] = self.ndcg_at_k(recommended, ground_truth, k)
            metrics[f'hit_rate@{k}'] = self.hit_rate_at_k(recommended, ground_truth, k)
        
        metrics['mrr'] = self.mrr(recommended, ground_truth)
        
        return metrics

# Test
if __name__ == "__main__":
    evaluator = RecommendationEvaluator()
    
    # Test con títulos exactos
    recommended = [
        "The Matrix (1999)",
        "Inception (2010)",
        "Blade Runner (1982)",
        "The Prestige (2006)"
    ]
    
    ground_truth = [
        "The Matrix (1999)",
        "Inception (2010)",
        "Interstellar (2014)"
    ]
    
    print("Testing evaluator:")
    print(f"Recommended: {recommended}")
    print(f"Ground truth: {ground_truth}")
    print()
    
    metrics = evaluator.evaluate_all(recommended, ground_truth, k_values=[3, 5])
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Test fuzzy matching
    print("\n\nTesting fuzzy matching:")
    test_pairs = [
        ("The Matrix (1999)", "Matrix (1999)"),
        ("The Lord of the Rings", "Lord of Rings"),
        ("Star Wars", "Star Wars: A New Hope"),
    ]
    
    for t1, t2 in test_pairs:
        similarity = evaluator.fuzzy_match(t1, t2)
        print(f"{t1} <-> {t2}: {similarity:.3f}")