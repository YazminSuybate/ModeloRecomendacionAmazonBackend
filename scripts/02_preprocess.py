"""Paso 2 (CRISP-DM, Fase 03): periodo → calidad → usuarios activos → codificación → partición.

1. Población: reseñas de Video_Games entre `period.start` y `period.end`.
2. Calidad: registros incompletos, una interacción por par usuario-ítem, ítems sin metadatos.
3. Muestra: usuarios con al menos `min_user_interactions` interacciones (los ítems no se filtran).
4. Partición temporal por usuario: test = 20 %; validación = 20 % más reciente del 80 % restante.

El tamaño de la muestra queda en `sample_report.json` (también en `reports/tables/`) para
contrastarlo con la tesis (3.1.5).
"""

import json

from recsys.config import load_config
from recsys.data.loader import raw_paths, read_jsonl
from recsys.data.preprocessing import (
    encode_ids,
    filter_active_users,
    filter_period,
    sample_report,
)
from recsys.data.quality import apply_quality_rules
from recsys.data.splitting import temporal_user_split
from recsys.data.storage import save_processed
from recsys.schema import ITEM_ID, ITEM_IDX, META_COLUMNS, REVIEW_COLUMNS
from recsys.utils.logging import get_logger

logger = get_logger("02_preprocess")


def main() -> None:
    config = load_config()
    reviews_path, meta_path = raw_paths(config.dataset, config.paths.raw)

    reviews = read_jsonl(reviews_path, REVIEW_COLUMNS)
    meta = read_jsonl(meta_path, META_COLUMNS)
    logger.info("Leídos: %d reseñas, %d productos", len(reviews), len(meta))

    reviews = filter_period(reviews, config.period.start, config.period.end)
    n_in_period = len(reviews)
    logger.info("Periodo %s a %s: %d reseñas", config.period.start, config.period.end, n_in_period)

    reviews, meta, report = apply_quality_rules(
        reviews, meta, config.quality.meta_completeness_fields, config.quality.min_completeness
    )
    logger.info("Reglas de calidad: %s", report)

    active = filter_active_users(reviews, config.filtering.min_user_interactions)
    sample = sample_report(active, n_in_period)
    logger.info("Muestra: %s", sample)

    encoded, users, items = encode_ids(active)
    item_table = (
        meta.set_index(ITEM_ID).loc[items].reset_index().assign(**{ITEM_IDX: range(len(items))})
    )
    train, val, test = temporal_user_split(
        encoded, config.split.val_ratio, config.split.test_ratio
    )
    logger.info("train=%d, val=%d, test=%d", len(train), len(val), len(test))

    processed = config.paths.processed
    save_processed({"train": train, "val": val, "test": test}, item_table, len(users), processed)
    (processed / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for directory in (processed, config.paths.reports / "tables"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "sample_report.json").write_text(json.dumps(sample, indent=2), "utf-8")
    logger.info("Guardado en %s", processed)


if __name__ == "__main__":
    main()
