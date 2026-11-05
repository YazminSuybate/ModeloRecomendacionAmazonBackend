"""R7: traduce puntajes de preferencia a ratings 1-5 para medir RMSE y MAE.

`implicit` no predice ratings: optimiza preferencias ponderadas por confianza. Esta capa ajusta
una regresión lineal de scikit-learn (puntaje del modelo, media del usuario, media del producto)
→ rating y recorta la salida al rango [1, 5].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression

from recsys.data.dataset import InteractionData
from recsys.schema import ITEM_IDX, RATING, USER_IDX

RATING_MIN, RATING_MAX = 1.0, 5.0


@dataclass(frozen=True)
class RatingBiases:
    """Medias de rating por usuario y por producto, suavizadas hacia la media global."""

    user_mean: np.ndarray
    item_mean: np.ndarray

    @classmethod
    def fit(cls, train: InteractionData, damping: float = 5.0) -> RatingBiases:
        frame = train.frame
        global_mean = frame[RATING].mean()

        def smoothed(index: str, size: int) -> np.ndarray:
            sums = np.bincount(frame[index], weights=frame[RATING], minlength=size)
            counts = np.bincount(frame[index], minlength=size)
            return (sums + damping * global_mean) / (counts + damping)

        return cls(smoothed(USER_IDX, train.n_users), smoothed(ITEM_IDX, train.n_items))

    def features(
        self, scores: np.ndarray, user_idx: np.ndarray, item_idx: np.ndarray
    ) -> np.ndarray:
        """Matriz de entrada del calibrador: (puntaje, media del usuario, media del producto)."""
        return np.column_stack([scores, self.user_mean[user_idx], self.item_mean[item_idx]])


class RatingCalibrator:
    def __init__(self) -> None:
        self._regression = LinearRegression()

    def fit(self, features: np.ndarray, ratings: np.ndarray) -> RatingCalibrator:
        self._regression.fit(features, ratings)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.clip(self._regression.predict(features), RATING_MIN, RATING_MAX)
