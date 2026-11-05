"""Texto de cada producto y su vectorización TF-IDF (scikit-learn) para la Etapa 1."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from recsys.config import TEXT_FIELDS, ContentConfig


def _flatten(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(part) for part in value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def build_item_text(meta: pd.DataFrame, fields: Sequence[str] = TEXT_FIELDS) -> pd.Series:
    """Concatena los atributos descriptivos de cada producto (I1 = len(fields)).

    Los campos faltantes se imputan con texto vacío (tratamiento de los campos con completitud
    menor al umbral).
    """
    return meta[list(fields)].map(_flatten).agg(" ".join, axis=1).str.lower()


def fit_tfidf(texts: pd.Series, config: ContentConfig) -> tuple[TfidfVectorizer, sp.csr_matrix]:
    """Vectores TF-IDF (L2) por ítem; I2 = tamaño del vocabulario aprendido."""
    vectorizer = TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        sublinear_tf=True,
        dtype=np.float32,
    )
    return vectorizer, vectorizer.fit_transform(texts)
