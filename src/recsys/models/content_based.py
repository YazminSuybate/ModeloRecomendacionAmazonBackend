"""VI-D1 / Etapa 1: filtrado basado en contenido (I1 atributos, I2 dimensión del vector).

Cada producto se representa con su vector TF-IDF (filas con norma L2 = 1). El perfil de un
usuario es la suma de los vectores de sus productos de train ponderada por rating, y el puntaje de
un producto es la similitud coseno entre ese perfil y el vector del producto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from recsys.config import ContentConfig
from recsys.data.dataset import InteractionData
from recsys.features.text_features import build_item_text, fit_tfidf
from recsys.models.base import Recommender
from recsys.models.calibration import RATING_MAX, RATING_MIN
from recsys.schema import ITEM_IDX, RATING

_PAIR_BATCH = 20_000


class ContentBasedRecommender(Recommender):
    name = "content_based"

    def __init__(self, config: ContentConfig, items: pd.DataFrame) -> None:
        """`items` es la tabla de productos con `ITEM_IDX` y los campos de texto."""
        super().__init__()
        self.config = config
        self.items = items.sort_values(ITEM_IDX)

    @property
    def item_vectors(self) -> sp.csr_matrix:
        """Vectores TF-IDF de los ítems (también los usa ILD@K, I12)."""
        return self._item_vectors

    @property
    def n_attributes(self) -> int:
        """I1: número de atributos de ítem incorporados."""
        return len(self.config.text_fields)

    @property
    def vector_dimension(self) -> int:
        """I2: dimensión del vector de características del ítem."""
        return self._item_vectors.shape[1]

    def _fit(self, train: InteractionData) -> None:
        texts = build_item_text(self.items, self.config.text_fields)
        self.vectorizer, self._item_vectors = fit_tfidf(texts, self.config)
        self._ratings = train.to_csr(train.frame[RATING].to_numpy(dtype=np.float32))
        self._interacted = train.to_csr()
        counts = np.asarray(self._interacted.sum(axis=1)).ravel()
        sums = np.asarray(self._ratings.sum(axis=1)).ravel()
        global_mean = float(train.frame[RATING].mean())
        self._user_mean = np.divide(
            sums, counts, out=np.full_like(sums, global_mean), where=counts > 0
        )

    def user_profiles(self, user_idx: np.ndarray) -> sp.csr_matrix:
        """Perfiles TF-IDF de los usuarios (norma L2 = 1; vacíos si no tienen historial)."""
        return normalize(self._ratings[np.asarray(user_idx)] @ self._item_vectors)

    def _score_items(self, user_idx: np.ndarray) -> np.ndarray:
        # Disperso × denso es más rápido que disperso × disperso con perfiles tan densos.
        profiles = self.user_profiles(user_idx).toarray()
        return np.ascontiguousarray((self._item_vectors @ profiles.T).T, dtype=np.float32)

    def predict_rating(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        """Media de los ratings del usuario ponderada por la similitud de contenido.

        r(u, i) = sum_j sim(i, j) * r(u, j) / sum_j sim(i, j), con j en el historial de train
        del usuario. Sin similitud con su historial se usa la media del usuario.
        """
        user_idx, item_idx = np.asarray(user_idx), np.asarray(item_idx)
        predictions = np.empty(len(user_idx), dtype=np.float64)
        for start in range(0, len(user_idx), _PAIR_BATCH):
            users = user_idx[start : start + _PAIR_BATCH]
            targets = self._item_vectors[item_idx[start : start + _PAIR_BATCH]]
            weighted = _row_dot(self._ratings[users] @ self._item_vectors, targets)
            total = _row_dot(self._interacted[users] @ self._item_vectors, targets)
            predictions[start : start + _PAIR_BATCH] = np.divide(
                weighted, total, out=self._user_mean[users].astype(np.float64), where=total > 0
            )
        return np.clip(predictions, RATING_MIN, RATING_MAX)


def _row_dot(left: sp.csr_matrix, right: sp.csr_matrix) -> np.ndarray:
    """Producto punto fila a fila de dos matrices dispersas con la misma forma."""
    return np.asarray(left.multiply(right).sum(axis=1)).ravel()
