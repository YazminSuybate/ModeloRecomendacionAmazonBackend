"""Piezas compartidas por los scripts 03, 05 y 06: datos procesados y modelos entrenados."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from recsys.config import Config
from recsys.data.dataset import InteractionData
from recsys.data.storage import load_items, load_splits, train_with_validation
from recsys.models.als import ALSRecommender
from recsys.models.base import Recommender
from recsys.models.baselines import ItemMeanRecommender, PopularityRecommender
from recsys.models.content_based import ContentBasedRecommender
from recsys.models.hybrid import HybridRecommender

MODELS_FILE = "models.joblib"


@dataclass(frozen=True)
class ProcessedData:
    train: InteractionData  # train + validación: el 80 % de entrenamiento de la tesis
    test: InteractionData  # 20 % más reciente de cada usuario
    items: pd.DataFrame


def load_processed(config: Config) -> ProcessedData:
    splits = load_splits(config.paths.processed)
    return ProcessedData(
        train_with_validation(splits), splits["test"], load_items(config.paths.processed)
    )


def train_all(config: Config, data: ProcessedData) -> dict[str, Recommender]:
    """Entrena O1 (popularidad; media del ítem para RMSE/MAE) y O2 (híbrido) sobre el 80 %.

    `content_based` y `als` son los módulos del propio híbrido: así I6 compara el Top-K final con
    el del mismo ALS que lo compone, y cada etapa se reporta por separado.
    """
    hybrid = HybridRecommender(
        ContentBasedRecommender(config.content, data.items),
        ALSRecommender(config.als, config.seed),
        config.hybrid,
        config.content.n_candidates,
    ).fit(data.train)
    return {
        "popularity": PopularityRecommender().fit(data.train),
        "item_mean": ItemMeanRecommender().fit(data.train),
        "content_based": hybrid.content,
        "als": hybrid.als,
        "hybrid": hybrid,
    }


def save_models(models: dict[str, Recommender], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODELS_FILE
    joblib.dump(models, path)
    return path


def load_models(directory: Path) -> dict[str, Recommender]:
    path = directory / MODELS_FILE
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}; ejecuta primero scripts/03_train.py")
    return joblib.load(path)
