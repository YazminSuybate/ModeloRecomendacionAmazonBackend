"""VI-D3: sistema híbrido en dos etapas, medición O2 (Post-Test). `blend_weight` es I5.

Etapa 1: `ContentBasedRecommender` elige los `n_candidates` productos no vistos más parecidos al
perfil del usuario.
Etapa 2: `ALSRecommender` puntúa esos candidatos; el puntaje final es
`blend_weight * ALS + (1 - blend_weight) * contenido`, con ambos puntajes llevados a [0, 1]
(min-max entre los candidatos de cada usuario). Los productos fuera de los candidatos reciben
-inf y nunca se recomiendan.
"""

from __future__ import annotations

import numpy as np

from recsys.config import HybridConfig
from recsys.data.dataset import InteractionData
from recsys.models.als import ALSRecommender
from recsys.models.base import Recommender, top_k
from recsys.models.content_based import ContentBasedRecommender


class HybridRecommender(Recommender):
    name = "hybrid"

    def __init__(
        self,
        content: ContentBasedRecommender,
        als: ALSRecommender,
        config: HybridConfig,
        n_candidates: int,
    ) -> None:
        super().__init__()
        self.content = content
        self.als = als
        self.config = config
        self.n_candidates = n_candidates

    def _fit(self, train: InteractionData) -> None:
        self.content.fit(train)
        self.als.fit(train)

    def _score_items(self, user_idx: np.ndarray) -> np.ndarray:
        content_scores = self.content._score_items(user_idx)
        candidates = top_k(content_scores, self.n_candidates, self._seen[user_idx])
        als_scores = self.als._score_items(user_idx)
        blended = blend_candidates(content_scores, als_scores, candidates, self.config.blend_weight)

        scores = np.full(content_scores.shape, -np.inf, dtype=np.float32)
        np.put_along_axis(scores, candidates, blended, axis=1)
        return scores

    def stage_scores(
        self, user_idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Candidatos y puntajes normalizados de cada etapa: (candidatos, contenido, ALS, final).

        Lo usa el prototipo visual para mostrar cómo ALS reordena los candidatos del contenido.
        """
        content_scores = self.content._score_items(user_idx)
        candidates = top_k(content_scores, self.n_candidates, self._seen[user_idx])
        als_scores = self.als._score_items(user_idx)
        content_part = _min_max(np.take_along_axis(content_scores, candidates, axis=1))
        als_part = _min_max(np.take_along_axis(als_scores, candidates, axis=1))
        weight = self.config.blend_weight
        return candidates, content_part, als_part, weight * als_part + (1 - weight) * content_part

    def predict_rating(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        """Mezcla de los ratings estimados por cada etapa, con el mismo peso que el ranking."""
        weight = self.config.blend_weight
        als = self.als.predict_rating(user_idx, item_idx)
        content = self.content.predict_rating(user_idx, item_idx)
        return weight * als + (1 - weight) * content


def blend_candidates(
    content_scores: np.ndarray, als_scores: np.ndarray, candidates: np.ndarray, weight: float
) -> np.ndarray:
    """Puntaje final de cada candidato: `weight * ALS + (1 - weight) * contenido` en [0, 1]."""
    content_part = _min_max(np.take_along_axis(content_scores, candidates, axis=1))
    als_part = _min_max(np.take_along_axis(als_scores, candidates, axis=1))
    return weight * als_part + (1 - weight) * content_part


def _min_max(scores: np.ndarray) -> np.ndarray:
    """Lleva cada fila a [0, 1]; una fila constante queda en 0."""
    low = scores.min(axis=1, keepdims=True)
    span = scores.max(axis=1, keepdims=True) - low
    return np.divide(scores - low, span, out=np.zeros_like(scores), where=span > 0)
