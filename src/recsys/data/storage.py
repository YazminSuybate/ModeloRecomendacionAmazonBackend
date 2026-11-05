"""Persistencia de los datos procesados (particiones, productos y dimensiones)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recsys.data.dataset import InteractionData
from recsys.utils.io import load_parquet, save_parquet

SPLITS = ("train", "val", "test")
_SHAPE_FILE = "shape.json"


def save_processed(
    splits: dict[str, pd.DataFrame], items: pd.DataFrame, n_users: int, directory: Path
) -> None:
    for name in SPLITS:
        save_parquet(splits[name], directory / f"{name}.parquet")
    save_parquet(items, directory / "items.parquet")
    shape = {"n_users": n_users, "n_items": len(items)}
    (directory / _SHAPE_FILE).write_text(json.dumps(shape, indent=2), encoding="utf-8")


def load_splits(directory: Path) -> dict[str, InteractionData]:
    shape = json.loads((directory / _SHAPE_FILE).read_text(encoding="utf-8"))
    return {
        name: InteractionData(load_parquet(directory / f"{name}.parquet"), **shape)
        for name in SPLITS
    }


def train_with_validation(splits: dict[str, InteractionData]) -> InteractionData:
    """Train + validación: con esto se reentrena cada modelo antes de medirlo en test."""
    train, val = splits["train"], splits["val"]
    frame = pd.concat([train.frame, val.frame], ignore_index=True)
    return InteractionData(frame, n_users=train.n_users, n_items=train.n_items)


def load_items(directory: Path) -> pd.DataFrame:
    return load_parquet(directory / "items.parquet")
