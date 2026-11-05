"""Paso 3 (CRISP-DM, Fase 04): entrena O1 y el sistema híbrido y registra la VI.

Cada modelo se entrena con train + validación (el 80 % de la tesis) usando los hiperparámetros
elegidos por 04_tune y copiados en `configs/config.yaml`.

Guarda los modelos en `artifacts/models/` y la configuración efectiva en
`reports/tables/vi_parameters.csv`: es el Instrumento 02 (registro de hiperparámetros) y alimenta
la ficha de verificación de componentes (Instrumento 01) con I1-I5. I6 se calcula en 05_evaluate.
"""

import pandas as pd

from recsys.config import load_config
from recsys.models.hybrid import HybridRecommender
from recsys.pipeline import load_processed, save_models, train_all
from recsys.utils.logging import get_logger
from recsys.utils.seed import set_seed

logger = get_logger("03_train")


def vi_parameters(hybrid: HybridRecommender, seed: int) -> pd.DataFrame:
    content, als = hybrid.content, hybrid.als
    rows = [
        ("D1", "I1", "Número de atributos de ítem incorporados", content.n_attributes,
         ", ".join(content.config.text_fields)),
        ("D1", "I2", "Dimensión del vector de características del ítem", content.vector_dimension,
         f"TF-IDF, ngram_range={content.config.ngram_range}, min_df={content.config.min_df}"),
        ("D2", "I3", "Número de factores latentes", als.config.factors, "implicit ALS"),
        ("D2", "I4", "Coeficiente de regularización", als.config.regularization, "implicit ALS"),
        ("D3", "I5", "Peso de combinación (peso de ALS)", hybrid.config.blend_weight,
         "w · ALS + (1 − w) · contenido"),
        ("—", "—", "Candidatos de la Etapa 1", hybrid.n_candidates, ""),
        ("—", "—", "Iteraciones de ALS", als.config.iterations, ""),
        ("—", "—", "Alpha (confianza) de ALS", als.config.alpha, ""),
        ("—", "—", "Semilla", seed, ""),
        ("—", "—", "Tiempo de factorización ALS (s)", als.factorization_time_s,
         "validación técnica"),
        ("—", "—", "Tiempo ALS con calibración de ratings (s)", als.train_time_s, ""),
        ("—", "—", "Tiempo de entrenamiento contenido (s)", content.train_time_s, ""),
        ("—", "—", "Tiempo de entrenamiento híbrido total (s)", hybrid.train_time_s, ""),
    ]
    return pd.DataFrame(rows, columns=["dimension", "indicator", "name", "value", "detail"])


def main() -> None:
    config = load_config()
    set_seed(config.seed)
    data = load_processed(config)

    models = train_all(config, data)
    for name, model in models.items():
        logger.info("%s entrenado en %.2f s", name, model.train_time_s)
    logger.info("Modelos en %s", save_models(models, config.paths.models))

    output = config.paths.tables / "vi_parameters.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    vi_parameters(models["hybrid"], config.seed).to_csv(output, index=False)
    logger.info("Parámetros de la VI en %s", output)


if __name__ == "__main__":
    main()
