"""Población (periodo), muestra (usuarios activos) y codificación de identificadores."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from recsys.schema import ITEM_ID, ITEM_IDX, TIMESTAMP, USER_ID, USER_IDX


def _epoch_ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)


def filter_period(interactions: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Conserva las interacciones registradas entre `start` y `end`, ambos días incluidos (UTC).

    Amazon Reviews 2023 guarda `timestamp` en milisegundos desde 1970.
    """
    lower, upper = _epoch_ms(start), _epoch_ms(end + timedelta(days=1))
    timestamps = interactions[TIMESTAMP].astype("int64")
    return interactions[(timestamps >= lower) & (timestamps < upper)].reset_index(drop=True)


def filter_active_users(
    interactions: pd.DataFrame, min_interactions: int, user_col: str = USER_ID
) -> pd.DataFrame:
    """Conserva los usuarios con al menos `min_interactions` interacciones.

    Solo se filtran usuarios, no productos: así lo define la muestra de la tesis. Debe aplicarse
    después de eliminar duplicados (una interacción por par usuario-ítem).
    """
    counts = interactions[user_col].map(interactions[user_col].value_counts())
    return interactions[counts >= min_interactions].reset_index(drop=True)


def encode_ids(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index, pd.Index]:
    """Asigna índices enteros consecutivos a usuarios y productos.

    Devuelve las interacciones con `USER_IDX`/`ITEM_IDX` y los índices para revertir el mapeo.
    """
    user_codes, users = pd.factorize(interactions[USER_ID], sort=True)
    item_codes, items = pd.factorize(interactions[ITEM_ID], sort=True)
    encoded = interactions.assign(**{USER_IDX: user_codes, ITEM_IDX: item_codes})
    return encoded, pd.Index(users, name=USER_ID), pd.Index(items, name=ITEM_ID)


def sample_report(interactions: pd.DataFrame, n_reviews_in_period: int) -> dict[str, float]:
    """Tamaño de la muestra tal como lo reporta la tesis (3.1.5)."""
    n_users = interactions[USER_ID].nunique()
    n_items = interactions[ITEM_ID].nunique()
    n_interactions = len(interactions)
    return {
        "reviews_in_period": n_reviews_in_period,
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "density_pct": round(100 * n_interactions / (n_users * n_items), 4) if n_items else 0.0,
    }
