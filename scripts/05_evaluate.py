"""Paso 5 (CRISP-DM, Fase 05): Pre-Test (O1) vs. Post-Test (O2) sobre el 20 % de test.

Requiere los modelos de 03_train, entrenados con train + validación (el 80 % de la tesis).
Escribe en `reports/tables/`:

- `results.csv`: promedios de I7-I12 (y F1@K, RMSE, MAE, tiempo) de cada modelo.
- `per_user_metrics.csv`: I7-I10 e I12 por usuario para O1 (popularidad) y O2 (híbrido).
- `hypothesis_tests.csv`: Shapiro-Wilk + t pareada / Wilcoxon por indicador y K (HE1-HE3).
- `pre_post.csv`: tabla O1 vs. O2 de los promedios, con la dirección de mejora.
- `hybridization_rate.csv`: I6 (Top-K del híbrido vs. Top-K de ALS).
- `technical_validation.csv`: RMSE y MAE frente a la media del ítem, y tiempo de ALS.
"""

import pandas as pd

from recsys.config import load_config
from recsys.evaluation.evaluator import evaluate_with_details
from recsys.evaluation.hybridization import hybridization_rate
from recsys.experiments.hypothesis_tests import hypothesis_table
from recsys.experiments.pre_post import pre_post_table
from recsys.pipeline import load_models, load_processed
from recsys.schema import USER_IDX
from recsys.utils.logging import get_logger
from recsys.utils.seed import set_seed

logger = get_logger("05_evaluate")

PRE_TEST, POST_TEST, RATING_BASELINE = "popularity", "hybrid", "item_mean"


def main() -> None:
    config = load_config()
    set_seed(config.seed)
    data = load_processed(config)
    models = load_models(config.paths.models)
    item_vectors = models[POST_TEST].content.item_vectors  # mismos vectores para todo ILD@K

    results = {
        name: evaluate_with_details(model, data.test, config.evaluation, item_vectors)
        for name, model in models.items()
    }
    for name, result in results.items():
        logger.info("%s: %s", name, result.summary)

    output = config.paths.tables
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.summary for result in results.values()]).to_csv(
        output / "results.csv", index=False
    )

    pre, post = results[PRE_TEST], results[POST_TEST]
    pd.concat(
        [pre.per_user.assign(model=PRE_TEST), post.per_user.assign(model=POST_TEST)]
    ).to_csv(output / "per_user_metrics.csv", index=False)

    tests = hypothesis_table(pre.per_user, post.per_user, config.statistics, USER_IDX)
    tests.to_csv(output / "hypothesis_tests.csv", index=False)
    for row in tests.itertuples():
        logger.info("%s %s: %s p=%.4g mejora=%.1f%% meta=%s", row.hypothesis, row.metric,
                    row.test, row.p_value, 100 * row.relative_improvement, row.meets_goal)

    pre_post_table(pre.summary, post.summary).to_csv(output / "pre_post.csv", index=False)

    als = results["als"]
    pd.DataFrame(
        [
            {
                "k": k,
                "indicator": "I6",
                "hybridization_rate": float(
                    hybridization_rate(post.recommended, als.recommended, k).mean()
                ),
            }
            for k in config.evaluation.k_values
        ]
    ).to_csv(output / "hybridization_rate.csv", index=False)

    baseline = results[RATING_BASELINE].summary
    pd.DataFrame(
        [
            {"metric": "rmse", "reference": baseline["rmse"], "hybrid": post.summary["rmse"]},
            {"metric": "mae", "reference": baseline["mae"], "hybrid": post.summary["mae"]},
            {"metric": "train_time_als_s", "reference": baseline["train_time_s"],
             "hybrid": models["als"].factorization_time_s},
        ]
    ).to_csv(output / "technical_validation.csv", index=False)
    logger.info("Resultados en %s", output)


if __name__ == "__main__":
    main()
