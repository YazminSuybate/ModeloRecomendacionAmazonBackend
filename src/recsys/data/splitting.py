"""Partición temporal por usuario: train-test 80/20 y validación tomada del 80 % de train."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.schema import TIMESTAMP, USER_IDX


def temporal_holdout(
    interactions: pd.DataFrame, holdout_ratio: float, user_col: str = USER_IDX
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa, por usuario, sus interacciones más recientes (`holdout_ratio`) de las anteriores.

    Cada usuario aporta al menos una interacción al bloque reciente (redondeo hacia abajo, mínimo
    1) y conserva al menos una en el anterior. Nunca se usa información del futuro.
    """
    frame = interactions.sort_values([user_col, TIMESTAMP], kind="stable")
    groups = frame.groupby(user_col)
    position_from_end = groups.cumcount(ascending=False).to_numpy()
    sizes = groups[user_col].transform("size").to_numpy()

    n_recent = np.clip(np.floor(sizes * holdout_ratio), 1, np.maximum(sizes - 1, 1)).astype(int)
    is_recent = position_from_end < n_recent
    return frame[~is_recent].reset_index(drop=True), frame[is_recent].reset_index(drop=True)


def temporal_user_split(
    interactions: pd.DataFrame, val_ratio: float, test_ratio: float, user_col: str = USER_IDX
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train-test 80/20 de la tesis, con la validación del ajuste sacada del 80 % de train.

    1. test = el `test_ratio` más reciente de cada usuario (20 %).
    2. val = el `val_ratio` más reciente de lo que queda; train = el resto.

    train + val es el conjunto de entrenamiento de la tesis (80 %): con él se reentrena cada
    modelo antes de medirlo en test (`storage.train_with_validation`).
    """
    train_full, test = temporal_holdout(interactions, test_ratio, user_col)
    train, val = temporal_holdout(train_full, val_ratio, user_col)
    return train, val, test
