"""Paso 4: búsqueda de hiperparámetros evaluando solo sobre validación.

1. ALS: entrena cada combinación de la grilla con train y la mide en validación.
2. Híbrido: con el mejor ALS y el modelo de contenido ya entrenados, prueba cada combinación de
   `n_candidates` × `blend_weight`. Los puntajes de cada etapa se calculan una sola vez por lote
   de usuarios y se reutilizan en toda la grilla.

Guarda las tablas en `reports/tables/tuning_*.csv` y la mejor combinación en
`reports/tables/best_params.json`. No modifica `config.yaml`: los valores se copian a mano.
"""

import dataclasses
import itertools
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from recsys.config import Config, load_config
from recsys.data.dataset import InteractionData
from recsys.data.storage import load_items, load_splits
from recsys.evaluation.coverage import catalog_coverage
from recsys.evaluation.evaluator import evaluate
from recsys.evaluation.ranking_metrics import per_user_ranking_metrics, ranking_scores_at_k
from recsys.evaluation.rating_metrics import mae, rmse
from recsys.models.als import ALSRecommender
from recsys.models.base import top_k
from recsys.models.content_based import ContentBasedRecommender
from recsys.models.hybrid import blend_candidates
from recsys.schema import ITEM_IDX, RATING, USER_IDX
from recsys.utils.logging import get_logger
from recsys.utils.seed import set_seed

logger = get_logger("04_tune")


def tune_als(config: Config, train: InteractionData, val: InteractionData) -> pd.DataFrame:
    grid = itertools.product(
        config.tuning.als_factors, config.tuning.als_regularization, config.tuning.als_alpha
    )
    rows = []
    for factors, regularization, alpha in grid:
        als_config = dataclasses.replace(
            config.als, factors=factors, regularization=regularization, alpha=alpha
        )
        model = ALSRecommender(als_config, config.seed).fit(train)
        result = evaluate(model, val, config.evaluation)
        rows.append(
            {"factors": factors, "regularization": regularization, "alpha": alpha, **result}
        )
        logger.info(
            "ALS %s: %s=%.4f", als_config, config.tuning.metric, result[config.tuning.metric]
        )
    return pd.DataFrame(rows).drop(columns="model")


def tune_hybrid(
    config: Config,
    content: ContentBasedRecommender,
    als: ALSRecommender,
    train: InteractionData,
    val: InteractionData,
) -> pd.DataFrame:
    tuning, evaluation = config.tuning, config.evaluation
    frame = val.frame
    relevant = (
        frame[frame[RATING] >= evaluation.relevance_threshold].groupby(USER_IDX)[ITEM_IDX].agg(set)
    )
    users = relevant.index.to_numpy()
    seen = train.to_csr()
    k_max, n_max = max(evaluation.k_values), max(tuning.n_candidates)
    grid = list(itertools.product(tuning.n_candidates, tuning.blend_weights))
    recommended: dict[tuple[int, float], list[np.ndarray]] = {key: [] for key in grid}

    for start in tqdm(range(0, len(users), evaluation.batch_size), desc="híbrido", unit="lote"):
        batch = users[start : start + evaluation.batch_size]
        content_scores = content._score_items(batch)
        als_scores = als._score_items(batch)
        # top_k ordena de mayor a menor: los n primeros de la lista más larga son los candidatos
        # para cualquier n más chico.
        all_candidates = top_k(content_scores, n_max, seen[batch])
        for n_candidates, weight in grid:
            candidates = all_candidates[:, :n_candidates]
            blended = blend_candidates(content_scores, als_scores, candidates, weight)
            order = np.argsort(-blended, axis=1, kind="stable")[:, :k_max]
            recommended[n_candidates, weight].append(np.take_along_axis(candidates, order, 1))

    pairs_users, pairs_items = frame[USER_IDX].to_numpy(), frame[ITEM_IDX].to_numpy()
    als_ratings = als.predict_rating(pairs_users, pairs_items)
    content_ratings = content.predict_rating(pairs_users, pairs_items)
    ratings = frame[RATING].to_numpy()

    rows = []
    for (n_candidates, weight), batches in recommended.items():
        recs = np.vstack(batches)
        predicted = weight * als_ratings + (1 - weight) * content_ratings
        row = {
            "n_candidates": n_candidates,
            "blend_weight": weight,
            "rmse": rmse(ratings, predicted),
            "mae": mae(ratings, predicted),
        }
        for k in evaluation.k_values:
            scores = ranking_scores_at_k(recs, relevant.to_list(), k)
            per_user = per_user_ranking_metrics(recs, relevant.to_list(), k)
            row |= {f"precision@{k}": scores.precision, f"recall@{k}": scores.recall}
            row |= {
                f"ndcg@{k}": float(per_user["ndcg"].mean()),
                f"mrr@{k}": float(per_user["mrr"].mean()),
                f"f1@{k}": scores.f1,
                f"coverage@{k}": catalog_coverage(recs[:, :k], val.n_items),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    set_seed(config.seed)
    splits = load_splits(config.paths.processed)
    train, val = splits["train"], splits["val"]
    metric = config.tuning.metric
    tables = config.paths.reports / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    als_table = tune_als(config, train, val)
    als_table.to_csv(tables / "tuning_als.csv", index=False)
    best_als = als_table.loc[als_table[metric].idxmax()]
    logger.info("Mejor ALS: %s", best_als[["factors", "regularization", "alpha", metric]].to_dict())

    als_config = dataclasses.replace(
        config.als,
        factors=int(best_als["factors"]),
        regularization=float(best_als["regularization"]),
        alpha=float(best_als["alpha"]),
    )
    content = ContentBasedRecommender(config.content, load_items(config.paths.processed))
    # Train y val salen del 80 % de entrenamiento: test (20 %) no participa en el ajuste.
    als = ALSRecommender(als_config, config.seed)
    content.fit(train)
    als.fit(train)

    hybrid_table = tune_hybrid(config, content, als, train, val)
    hybrid_table.to_csv(tables / "tuning_hybrid.csv", index=False)
    best_hybrid = hybrid_table.loc[hybrid_table[metric].idxmax()]
    logger.info(
        "Mejor híbrido: %s", best_hybrid[["n_candidates", "blend_weight", metric]].to_dict()
    )

    best = {
        "metric": metric,
        "als": {
            "factors": als_config.factors,
            "regularization": als_config.regularization,
            "alpha": als_config.alpha,
        },
        "hybrid": {
            "n_candidates": int(best_hybrid["n_candidates"]),
            "blend_weight": float(best_hybrid["blend_weight"]),
        },
        "validation": {"als": float(best_als[metric]), "hybrid": float(best_hybrid[metric])},
    }
    (tables / "best_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    logger.info("Tablas de ajuste en %s", tables)


if __name__ == "__main__":
    main()
