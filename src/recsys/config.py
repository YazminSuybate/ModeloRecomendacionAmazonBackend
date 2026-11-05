"""Carga y validación de `configs/config.yaml`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

MIN_INTERACTIONS = 5  # Muestra de la tesis: usuarios con al menos 5 interacciones
ALTERNATIVES = ("greater", "two-sided")
TEXT_FIELDS = ("title", "features", "description", "categories", "store")


@dataclass(frozen=True)
class PathsConfig:
    raw: Path
    interim: Path
    processed: Path
    reports: Path
    artifacts: Path

    @property
    def models(self) -> Path:
        return self.artifacts / "models"

    @property
    def tables(self) -> Path:
        return self.reports / "tables"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    category: str
    reviews_url: str
    meta_url: str


@dataclass(frozen=True)
class PeriodConfig:
    """Periodo de la población; ambas fechas se incluyen."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"period.start ({self.start}) es posterior a period.end ({self.end})")


@dataclass(frozen=True)
class QualityConfig:
    min_completeness: float
    meta_completeness_fields: tuple[str, ...]


@dataclass(frozen=True)
class FilteringConfig:
    min_user_interactions: int

    def __post_init__(self) -> None:
        if self.min_user_interactions < MIN_INTERACTIONS:
            raise ValueError(
                f"La tesis exige min_user_interactions >= {MIN_INTERACTIONS}, "
                f"se recibió {self.min_user_interactions}"
            )


@dataclass(frozen=True)
class SplitConfig:
    test_ratio: float  # 20 % de cada usuario
    val_ratio: float  # fracción del 80 % de train que se usa como validación del ajuste

    def __post_init__(self) -> None:
        for name, ratio in (("test_ratio", self.test_ratio), ("val_ratio", self.val_ratio)):
            if not 0 < ratio < 1:
                raise ValueError(f"{name} debe estar entre 0 y 1, se recibió {ratio}")


@dataclass(frozen=True)
class EvaluationConfig:
    k_values: tuple[int, ...]
    relevance_threshold: float
    batch_size: int


@dataclass(frozen=True)
class StatisticsConfig:
    alpha: float
    alternative: str
    min_relative_improvement: float

    def __post_init__(self) -> None:
        if self.alternative not in ALTERNATIVES:
            raise ValueError(f"statistics.alternative debe ser uno de {ALTERNATIVES}")


@dataclass(frozen=True)
class ContentConfig:
    max_features: int
    ngram_range: tuple[int, int]
    min_df: int
    n_candidates: int
    text_fields: tuple[str, ...] = TEXT_FIELDS  # I1 = número de atributos


@dataclass(frozen=True)
class ALSConfig:
    factors: int
    regularization: float
    iterations: int
    alpha: float


@dataclass(frozen=True)
class HybridConfig:
    blend_weight: float


@dataclass(frozen=True)
class TuningConfig:
    """Grillas que `04_tune` recorre sobre validación; `metric` decide la mejor combinación."""

    metric: str
    als_factors: tuple[int, ...]
    als_regularization: tuple[float, ...]
    als_alpha: tuple[float, ...]
    n_candidates: tuple[int, ...]
    blend_weights: tuple[float, ...]


@dataclass(frozen=True)
class DemoConfig:
    n_users: int
    top_k: int


@dataclass(frozen=True)
class Config:
    seed: int
    paths: PathsConfig
    dataset: DatasetConfig
    period: PeriodConfig
    quality: QualityConfig
    filtering: FilteringConfig
    split: SplitConfig
    evaluation: EvaluationConfig
    statistics: StatisticsConfig
    content: ContentConfig
    als: ALSConfig
    hybrid: HybridConfig
    tuning: TuningConfig
    demo: DemoConfig


def _date(value: Any) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    quality, evaluation, content = raw["quality"], raw["evaluation"], raw["content"]
    return Config(
        seed=raw["seed"],
        paths=PathsConfig(**{key: PROJECT_ROOT / value for key, value in raw["paths"].items()}),
        dataset=DatasetConfig(**raw["dataset"]),
        period=PeriodConfig(start=_date(raw["period"]["start"]), end=_date(raw["period"]["end"])),
        quality=QualityConfig(
            min_completeness=quality["min_completeness"],
            meta_completeness_fields=tuple(quality["meta_completeness_fields"]),
        ),
        filtering=FilteringConfig(**raw["filtering"]),
        split=SplitConfig(**raw["split"]),
        evaluation=EvaluationConfig(
            k_values=tuple(evaluation["k_values"]),
            relevance_threshold=evaluation["relevance_threshold"],
            batch_size=evaluation["batch_size"],
        ),
        statistics=StatisticsConfig(**raw["statistics"]),
        content=ContentConfig(
            **{
                **content,
                "ngram_range": tuple(content["ngram_range"]),
                "text_fields": tuple(content.get("text_fields", TEXT_FIELDS)),
            }
        ),
        als=ALSConfig(**raw["als"]),
        hybrid=HybridConfig(**raw["hybrid"]),
        tuning=TuningConfig(
            **{key: tuple(v) if isinstance(v, list) else v for key, v in raw["tuning"].items()}
        ),
        demo=DemoConfig(**raw["demo"]),
    )
