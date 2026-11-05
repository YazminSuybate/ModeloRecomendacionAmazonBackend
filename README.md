# Sistema de recomendación híbrido para Amazon (contenido + ALS)

Prototipo de la tesis *"Sistema de recomendación híbrido con aprendizaje automático para mejorar la personalización de productos de Amazon, 2022-2023"* (UTP, Ingeniería de Software). Recomienda en dos etapas sobre **Amazon Reviews 2023** (Hou et al., 2024), categoría Video_Games:

1. **Etapa 1, filtrado basado en contenido:** TF-IDF de los atributos de cada producto (scikit-learn) para elegir candidatos.
2. **Etapa 2, factorización de matrices con ALS:** la librería `implicit` puntúa esos candidatos, combinando interacciones implícitas y explícitas, y se mezcla con el puntaje de contenido.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
```

## Pipeline (CRISP-DM)

| Paso | Script | Fase CRISP-DM | Salida en `reports/tables/` |
|---|---|---|---|
| 1. Descarga | `python scripts/01_download.py` | 02 Comprensión de los datos | — |
| 2. Periodo → calidad → usuarios ≥ 5 → partición | `python scripts/02_preprocess.py` | 03 Preparación | `sample_report.json` |
| 3. Ajuste en validación (I3, I4, I5) | `python scripts/04_tune.py` | 04 Modelado | `tuning_als.csv`, `tuning_hybrid.csv`, `best_params.json` |
| 4. Entrenamiento con train + validación | `python scripts/03_train.py` | 04 Modelado | `vi_parameters.csv` (+ `artifacts/models/`) |
| 5. Pre-Test vs. Post-Test en test | `python scripts/05_evaluate.py` | 05 Evaluación | `results.csv`, `hypothesis_tests.csv`, `pre_post.csv`, `hybridization_rate.csv`, `technical_validation.csv`, `per_user_metrics.csv` |
| 6. Paquete del prototipo | `python scripts/06_export_demo.py` | 06 Despliegue | `artifacts/demo/` |

Orden: 01 → 02 → 04 → copiar `best_params.json` a `configs/config.yaml` → 03 → 05 → 06. Todos los parámetros están en [`configs/config.yaml`](configs/config.yaml).

## Trazabilidad con la Matriz de Consistencia

| Elemento de la matriz | Implementación | Salida | Test |
|---|---|---|---|
| Población: Video_Games, ene-2022 a sep-2023 | `data/preprocessing.py::filter_period` | `sample_report.json` | `tests/test_sample.py` |
| Muestra: usuarios con ≥ 5 interacciones, una por par usuario-ítem (los ítems no se filtran) | `filter_active_users`, `quality.py::drop_duplicate_interactions` | `sample_report.json` | `tests/test_sample.py` |
| Partición train-test 80/20 por usuario | `data/splitting.py::temporal_user_split` | `train/val/test.parquet` | `tests/test_splitting.py` |
| Relevante = calificación ≥ 4; solo usuarios con ≥ 1 relevante | `evaluation/evaluator.py` | — | `tests/test_baselines.py` |
| VI-D1 · I1 atributos de ítem, I2 dimensión del vector | `models/content_based.py`, `features/text_features.py` | `vi_parameters.csv` | `tests/test_content_based.py`, `tests/test_matrix_indicators.py` |
| VI-D2 · I3 factores latentes, I4 regularización | `models/als.py` | `vi_parameters.csv` | `tests/test_als.py` |
| VI-D3 · I5 peso de combinación | `models/hybrid.py` | `vi_parameters.csv`, `tuning_hybrid.csv` | `tests/test_hybrid.py` |
| VI-D3 · I6 tasa de hibridación del ranking | `evaluation/hybridization.py` | `hybridization_rate.csv` | `tests/test_metrics.py` |
| VD-D1 · I7 Precision@K, I8 Recall@K | `evaluation/ranking_metrics.py` | `results.csv`, `per_user_metrics.csv` | `tests/test_metrics.py` |
| VD-D2 · I9 NDCG@K, I10 MRR@K | `evaluation/ranking_metrics.py` | `results.csv`, `per_user_metrics.csv` | `tests/test_metrics.py` |
| VD-D3 · I11 Coverage@K (descriptivo) | `evaluation/coverage.py` | `results.csv` | `tests/test_metrics.py` |
| VD-D3 · I12 ILD@K | `evaluation/diversity.py` | `results.csv`, `per_user_metrics.csv` | `tests/test_metrics.py` |
| Pre-Test O1 = popularidad; Post-Test O2 = híbrido | `scripts/05_evaluate.py` | `pre_post.csv` | — |
| Shapiro-Wilk sobre diferencias → t pareada o Wilcoxon, α = 0.05; meta ≥ 10 % | `experiments/hypothesis_tests.py` | `hypothesis_tests.csv` | `tests/test_hypothesis.py` |
| Validación técnica: RMSE, MAE vs. media del ítem; tiempo de ALS | `models/calibration.py`, `models/als.py` | `technical_validation.csv` | `tests/test_als.py`, `tests/test_matrix_indicators.py` |
| Python + scikit-learn + Implicit; sin deep learning ni LLM | `requirements.txt` | — | `tests/test_scope.py` |

Hipótesis → indicadores contrastados: **HE1** I7, I8 · **HE2** I9, I10 · **HE3** I12 (I11 se reporta de forma descriptiva).

## Decisiones metodológicas

- **Orden de la Fase 03:** periodo → registros incompletos → una interacción por par usuario-ítem (la más reciente) → ítems sin metadatos → usuarios con ≥ 5 interacciones. `sample_report.json` permite compararlo con la tesis (6,869 usuarios, 15,710 ítems, 53,593 interacciones). **Aún no se ha verificado con los datos reales.**
- **Partición temporal por usuario:** test es el 20 % más reciente de cada usuario (mínimo 1, redondeo hacia abajo). La validación del ajuste es el 20 % más reciente del 80 % restante. Así, test es exactamente el 20 % del 80/20 de la tesis, y train + validación es su 80 %.
- **Reentrenamiento para test:** los hiperparámetros se eligen en validación; para la evaluación final, cada modelo (O1 y O2) se reentrena con train + validación y se mide en test. Con la muestra anterior, entrenar solo con train dejaba al modelo de contenido con R@10 = 0,0171 en test, frente a 0,0235 con train + validación; la popularidad casi no cambiaba (0,0276 frente a 0,0281).
- **Confianza de ALS:** preferencia 1 si la interacción existe (señal implícita) y confianza c = 1 + α · rating / 5 (señal explícita).
- **Híbrido:** el contenido elige `n_candidates` productos no vistos y el puntaje final es `blend_weight · ALS + (1 − blend_weight) · contenido`, ambos llevados a [0, 1] entre los candidatos. El rating estimado mezcla los de cada etapa con el mismo peso.
- **I6:** proporción del Top-K del híbrido que no está en el Top-K de su propio módulo ALS.
- **ILD@K:** 1 − coseno entre los vectores TF-IDF de contenido; los mismos vectores para O1 y O2.
- **Prueba estadística:** unilateral (`alternative: greater`), porque las hipótesis afirman una mejora. Wilcoxon descarta los pares con diferencia 0. Con n > 5000, Shapiro-Wilk da un p aproximado.
- **RMSE y MAE con ALS:** `implicit` no predice ratings, así que una regresión lineal (puntaje ALS, media del usuario, media del producto) calibra los puntajes a la escala de 1 a 5. Para no sesgarla, se aparta la última interacción de cada usuario dentro de train, se entrena un ALS auxiliar con el resto y la regresión se ajusta sobre esas interacciones apartadas. El tiempo de entrenamiento que se reporta (`factorization_time_s`) es solo el de la factorización.
- **Ajuste de hiperparámetros:** `04_tune.py` recorre en validación las grillas de `config.yaml → tuning` y elige por NDCG@10 (I9). F1@10 se sigue calculando, pero ya no es un indicador de la matriz.
- **Completitud de metadatos:** se mide por campo. Los campos que no llegan al 90 % (en Video_Games, `features` y `description`) se conservan, y sus valores faltantes se imputan con texto vacío. Ningún producto se elimina por completitud. Los porcentajes quedan en `quality_report.json`.

## Limitaciones

- **Cold start:** fuera del alcance, porque la muestra exige 5 interacciones por usuario. Los ítems con pocas interacciones sí se conservan; el módulo de contenido es el único que puede recomendarlos.
- **Diseño pre-experimental sin grupo de control:** los resultados no se generalizan más allá del dataset y la muestra.
- **Sin deep learning ni agentes LLM:** es una decisión de alcance.
- **Techo de la Etapa 1:** ALS solo reordena los candidatos que elige el contenido. Con la muestra anterior y 3.000 candidatos, el híbrido quedó un 3,6 % por debajo de ALS solo en F1@10, pero cubrió un 44 % más del catálogo.

## Resultados con la muestra anterior (referencia)

Obtenidos **antes** de alinear el pipeline con la matriz vigente (k-core de usuarios e ítems, sin filtro de periodo, partición 60/20/20). Hay que regenerarlos con 02 → 04 → 03 → 05; `results.csv` y las tablas de ajuste versionadas todavía son las anteriores.

| Modelo | RMSE | MAE | F1@10 | Cobertura@10 | Entrenamiento |
|---|---:|---:|---:|---:|---:|
| O1: popularidad | 1,263 | 0,982 | 0,0066 | 0,07 % | 0,007 s |
| O1: media por producto | 1,216 | 0,913 | 0,0002 | 0,05 % | 0,009 s |
| Etapa 1: contenido | 1,176 | 0,779 | 0,0053 | 81,9 % | 24,5 s |
| Etapa 2: ALS | 1,112 | 0,792 | 0,0193 | 15,2 % | 21,3 s |
| O2: híbrido | 1,112 | 0,787 | 0,0186 | 21,9 % | 46,5 s |

## Pendientes

1. **Regenerar resultados con la muestra nueva:** 02 → 04 → copiar `best_params.json` a `config.yaml` → 03 → 05, y comprobar que `sample_report.json` coincida con la tesis.
2. **Prototipo con datos reales:** correr `06_export_demo.py` y luego `make seed` para reemplazar los datos sintéticos.
3. **Notebook de exploración (EDA):** distribución de ratings, interacciones por usuario y por producto, evolución en el tiempo, largo de los textos y conteos de cada paso del preprocesamiento.
4. **Rutas de datos:** `config.yaml` apunta a `D:/TESIS-data`. Para reproducir en otra máquina, hay que cambiar las rutas o crear esa carpeta.
5. **Opcional, ampliar la grilla de ajuste:** 128 factores y 3.000 candidatos quedaron en el borde de la grilla anterior.

## Backend del prototipo visual

La interfaz vive en otro repo, **ModeloRecomendacionAmazonFrontend** (React). Este repo le sirve la API.

```
pipeline ML ──06_export_demo──▶ artifacts/demo/*.jsonl ──make seed──▶ Postgres ◀── FastAPI ◀── frontend (otro repo)
```

- `backend/`: FastAPI + SQLModel + Alembic, con capas routers → handlers → unit of work → repositorios. Solo lee recomendaciones precalculadas; no entrena modelos.
- `docker-compose.yml`: servicios `db`, `migrations`, `pgadmin` y `backend`.
- CORS: por defecto acepta `http://localhost:5173` y `http://localhost:5180`. Otros orígenes se agregan con `CORS_ORIGINS` en `.env`.

### Puesta en marcha

Requiere Docker Desktop.

```bash
cp .env.example .env
docker compose up -d --build
python scripts/dev_synthetic_demo.py      # o 06_export_demo.py con datos reales
docker compose exec backend uv run python -m src.seed
```

| Servicio | URL |
|---|---|
| API (Swagger) | http://localhost:8000/docs |
| pgAdmin | http://localhost:8080 |

`dev_synthetic_demo.py` genera **datos sintéticos** que solo sirven para desarrollar la interfaz. La API los marca con `isSynthetic` y el frontend muestra un aviso. Nunca deben usarse como resultados de la tesis.

### Comandos (`Makefile`)

En Windows sin `make`, usa el comando equivalente de la segunda columna.

| `make` | Equivalente |
|---|---|
| `make up` / `make down` | `docker compose up -d` / `docker compose down -v` |
| `make seed` | `docker compose exec backend uv run python -m src.seed` |
| `make add-migration name="x"` | `docker compose exec backend bash -c "uv run alembic revision --autogenerate -m 'x'"` |
| `make install-python name="pkg"` | `docker compose exec backend uv add pkg` y luego copiar `pyproject.toml` y `uv.lock` con `docker compose cp` |
| `make integration-test` | `docker compose exec backend uv run pytest src/test -v` |

`make integration-test` vacía las tablas del prototipo. Después de correrlo, vuelve a ejecutar `make seed`.

## Tests

```bash
pytest
```
