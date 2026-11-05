"""VD-D1 y VD-D2: exactitud (I7 Precision@K, I8 Recall@K) y calidad del ranking (I9 NDCG@K,
I10 MRR@K), calculadas por usuario para el contraste pareado Pre-Test / Post-Test."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankingScores:
    precision: float
    recall: float
    f1: float


def hit_matrix(recommended: np.ndarray, relevant: Sequence[set[int]], k: int) -> np.ndarray:
    """Matriz booleana (n_usuarios, k): True si el ítem en esa posición es relevante."""
    rows = [
        [item in rel for item in items[:k].tolist()]
        for items, rel in zip(recommended, relevant, strict=True)
    ]
    return np.array(rows, dtype=bool).reshape(len(relevant), k)


def per_user_ranking_metrics(
    recommended: np.ndarray, relevant: Sequence[set[int]], k: int
) -> dict[str, np.ndarray]:
    """Precision@K, Recall@K, NDCG@K y MRR@K de cada usuario (relevancia binaria).

    - Precision@K = |Rec_K(u) ∩ Rel(u)| / K
    - Recall@K = |Rec_K(u) ∩ Rel(u)| / |Rel(u)|
    - NDCG@K = DCG@K / IDCG@K, con DCG@K = Σ rel_i / log2(i + 1)
    - MRR@K: 1 / posición del primer relevante en el Top-K; 0 si no hay ninguno

    `recommended[u]` tiene al menos `k` productos y `relevant[u]` no está vacío.
    """
    n_users = len(relevant)
    if n_users == 0:
        empty = np.empty(0)
        return {"precision": empty, "recall": empty, "ndcg": empty, "mrr": empty}

    hits = hit_matrix(recommended, relevant, k)
    n_relevant = np.array([len(rel) for rel in relevant], dtype=float)
    n_hits = hits.sum(axis=1)

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * discounts).sum(axis=1)
    ideal_counts = np.minimum(n_relevant, k).astype(int)
    idcg = np.cumsum(discounts)[ideal_counts - 1]

    first_hit = np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, 0)
    mrr = np.divide(1.0, first_hit, out=np.zeros(n_users), where=first_hit > 0)

    return {
        "precision": n_hits / k,
        "recall": n_hits / n_relevant,
        "ndcg": dcg / idcg,
        "mrr": mrr,
    }


def ranking_scores_at_k(
    recommended: np.ndarray, relevant: Sequence[set[int]], k: int
) -> RankingScores:
    """Promedia Precision@K y Recall@K entre usuarios; F1@K es su media armónica.

    F1@K ya no es un indicador de la matriz; se conserva como dato complementario.
    """
    if len(relevant) == 0:
        return RankingScores(0.0, 0.0, 0.0)
    scores = per_user_ranking_metrics(recommended, relevant, k)
    precision = float(scores["precision"].mean())
    recall = float(scores["recall"].mean())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return RankingScores(precision, recall, f1)
