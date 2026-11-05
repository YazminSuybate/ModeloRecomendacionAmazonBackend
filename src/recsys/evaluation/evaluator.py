"""Protocolo único de evaluación: el mismo para el Pre-Test (O1) y el Post-Test (O2).

Solo se evalúan los usuarios con al menos un ítem relevante (calificación >= umbral) en test, y
se recomiendan ítems que el usuario no tiene en entrenamiento.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from recsys.config import EvaluationConfig
from recsys.data.dataset import InteractionData
from recsys.evaluation.coverage import catalog_coverage
from recsys.evaluation.diversity import intra_list_diversity
from recsys.evaluation.ranking_metrics import per_user_ranking_metrics
from recsys.evaluation.rating_metrics import mae, rmse
from recsys.models.base import Recommender
from recsys.schema import ITEM_IDX, RATING, USER_IDX

PER_USER_METRICS = ("precision", "recall", "ndcg", "mrr", "ild")  # I7, I8, I9, I10, I12


@dataclass(frozen=True)
class EvaluationResult:
    summary: dict[str, float | str]  # promedios por modelo (una fila de results.csv)
    per_user: pd.DataFrame  # una fila por usuario evaluado: insumo de las pruebas pareadas
    recommended: np.ndarray  # Top-max(K) de cada usuario evaluado, en el orden de `per_user`


def relevant_items(test: InteractionData, threshold: float) -> pd.Series:
    """Conjunto de ítems relevantes de test por usuario (solo usuarios con alguno)."""
    frame = test.frame
    return frame[frame[RATING] >= threshold].groupby(USER_IDX)[ITEM_IDX].agg(set).sort_index()


def recommend_in_batches(
    model: Recommender, users: np.ndarray, k: int, batch_size: int
) -> np.ndarray:
    if len(users) == 0:
        return np.empty((0, k), dtype=int)
    starts = tqdm(range(0, len(users), batch_size), desc=f"{model.name}: ranking", unit="lote")
    batches = [model.recommend(users[start : start + batch_size], k) for start in starts]
    return np.vstack(batches)


def evaluate_with_details(
    model: Recommender,
    test: InteractionData,
    config: EvaluationConfig,
    item_vectors: sp.csr_matrix | None = None,
) -> EvaluationResult:
    """Calcula I7-I12 (y F1@K complementario) por usuario y en promedio, más la validación
    técnica (RMSE, MAE, tiempo de entrenamiento). Sin `item_vectors` se omite ILD@K."""
    frame = test.frame
    predictions = model.predict_rating(frame[USER_IDX].to_numpy(), frame[ITEM_IDX].to_numpy())
    summary: dict[str, float | str] = {
        "model": model.name,
        "rmse": rmse(frame[RATING].to_numpy(), predictions),
        "mae": mae(frame[RATING].to_numpy(), predictions),
    }

    relevant = relevant_items(test, config.relevance_threshold)
    users = relevant.index.to_numpy()
    recommended = recommend_in_batches(model, users, max(config.k_values), config.batch_size)
    per_user = pd.DataFrame({USER_IDX: users})

    for k in config.k_values:
        scores = per_user_ranking_metrics(recommended, relevant.to_list(), k)
        for metric, values in scores.items():
            per_user[f"{metric}@{k}"] = values
        if item_vectors is not None and k >= 2:
            per_user[f"ild@{k}"] = intra_list_diversity(recommended, item_vectors, k)

        precision, recall = scores["precision"].mean(), scores["recall"].mean()
        for metric in PER_USER_METRICS:
            column = f"{metric}@{k}"
            if column in per_user:
                summary[column] = float(per_user[column].mean()) if len(users) else 0.0
        summary[f"f1@{k}"] = (
            0.0 if not len(users) or precision + recall == 0
            else float(2 * precision * recall / (precision + recall))
        )
        summary[f"coverage@{k}"] = catalog_coverage(recommended[:, :k], test.n_items)

    summary["n_users_evaluated"] = len(users)
    summary["train_time_s"] = model.train_time_s if model.train_time_s is not None else np.nan
    return EvaluationResult(summary, per_user, recommended)


def evaluate(
    model: Recommender,
    test: InteractionData,
    config: EvaluationConfig,
    item_vectors: sp.csr_matrix | None = None,
) -> dict[str, float | str]:
    """Promedios por modelo; ver `evaluate_with_details`."""
    return evaluate_with_details(model, test, config, item_vectors).summary
