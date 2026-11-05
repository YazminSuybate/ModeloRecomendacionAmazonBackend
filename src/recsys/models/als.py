"""VI-D2 / Etapa 2: factorización de matrices con ALS de la librería `implicit` (I3 factores
latentes, I4 regularización).

Combina las dos señales en la matriz usuarios × productos: la preferencia es 1 si la interacción
existe (implícita) y la confianza es c = 1 + alpha * rating / 5 (explícita). Las celdas vacías
tienen preferencia 0 y confianza 1.

RMSE y MAE (R7) necesitan ratings, y ALS solo da puntajes de preferencia. El calibrador se
ajusta sin mirar validación ni test: se aparta la última interacción de cada usuario dentro de
train, se entrena un ALS auxiliar con el resto y la regresión aprende a pasar de sus puntajes a
los ratings apartados.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from implicit.cpu.als import AlternatingLeastSquares
from threadpoolctl import threadpool_limits

from recsys.config import ALSConfig
from recsys.data.dataset import InteractionData
from recsys.models.base import Recommender
from recsys.models.calibration import RatingBiases, RatingCalibrator
from recsys.schema import ITEM_IDX, RATING, TIMESTAMP, USER_IDX
from recsys.utils.timing import Timer


class ALSRecommender(Recommender):
    name = "als"

    def __init__(self, config: ALSConfig, seed: int) -> None:
        super().__init__()
        self.config = config
        self.seed = seed

    def confidence(self, train: InteractionData) -> sp.csr_matrix:
        """Matriz de confianza c = 1 + alpha * rating / 5 sobre las interacciones observadas."""
        ratings = train.frame[RATING].to_numpy(dtype=np.float32)
        return train.to_csr(1.0 + self.config.alpha * ratings / 5.0)

    def _fit(self, train: InteractionData) -> None:
        with Timer() as timer:
            self._model = self._train_als(train)
        # Validación técnica: tiempo de la factorización, sin el ALS auxiliar del calibrador.
        self.factorization_time_s = timer.seconds
        self._biases = RatingBiases.fit(train)
        self._calibrator = self._fit_calibrator(train)

    def _train_als(self, train: InteractionData) -> AlternatingLeastSquares:
        # implicit ya paraleliza; BLAS con varios hilos lo vuelve más lento.
        with threadpool_limits(1, "blas"):
            model = AlternatingLeastSquares(
                factors=self.config.factors,
                regularization=self.config.regularization,
                iterations=self.config.iterations,
                random_state=self.seed,
            )
            model.fit(self.confidence(train), show_progress=False)
        return model

    def _fit_calibrator(self, train: InteractionData) -> RatingCalibrator:
        frame = train.frame.sort_values([USER_IDX, TIMESTAMP], kind="stable")
        is_last = ~frame.duplicated(USER_IDX, keep="last").to_numpy()
        rest = InteractionData(frame[~is_last], train.n_users, train.n_items)
        held_out = frame[is_last]

        model = self._train_als(rest)
        users, items = held_out[USER_IDX].to_numpy(), held_out[ITEM_IDX].to_numpy()
        scores = _pair_scores(model, users, items)
        features = RatingBiases.fit(rest).features(scores, users, items)
        return RatingCalibrator().fit(features, held_out[RATING].to_numpy())

    def _score_items(self, user_idx: np.ndarray) -> np.ndarray:
        return self._model.user_factors[user_idx] @ self._model.item_factors.T

    def predict_rating(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        user_idx, item_idx = np.asarray(user_idx), np.asarray(item_idx)
        scores = _pair_scores(self._model, user_idx, item_idx)
        return self._calibrator.predict(self._biases.features(scores, user_idx, item_idx))


def _pair_scores(
    model: AlternatingLeastSquares, user_idx: np.ndarray, item_idx: np.ndarray
) -> np.ndarray:
    """Puntaje ALS (producto punto de los factores) de cada par (usuario, producto)."""
    return np.einsum("ij,ij->i", model.user_factors[user_idx], model.item_factors[item_idx])
