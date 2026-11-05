"""Indicadores I6-I12 y validación técnica, verificados con valores calculados a mano."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from recsys.evaluation.coverage import catalog_coverage
from recsys.evaluation.diversity import intra_list_diversity
from recsys.evaluation.hybridization import hybridization_rate
from recsys.evaluation.ranking_metrics import per_user_ranking_metrics, ranking_scores_at_k
from recsys.evaluation.rating_metrics import mae, rmse
from recsys.experiments.pre_post import pre_post_table


def test_rmse_and_mae():
    y_true = np.array([5.0, 3.0, 4.0])
    y_pred = np.array([4.0, 3.0, 2.0])

    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(5 / 3))
    assert mae(y_true, y_pred) == pytest.approx(1.0)


def test_precision_recall_f1_at_k():
    recommended = np.array([[1, 2, 3], [4, 5, 6]])
    relevant = [{1, 9}, {7}]

    scores = ranking_scores_at_k(recommended, relevant, k=2)

    # Usuario 0: 1 acierto de 2 → P=0.5, R=0.5. Usuario 1: 0 aciertos.
    assert scores.precision == pytest.approx(0.25)
    assert scores.recall == pytest.approx(0.25)
    assert scores.f1 == pytest.approx(0.25)


def test_ndcg_and_mrr_per_user():
    recommended = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    relevant = [{1, 3}, {5}, {0}]

    scores = per_user_ranking_metrics(recommended, relevant, k=3)

    # Usuario 0: aciertos en 1 y 3 → DCG = 1 + 1/log2(4) = 1.5; IDCG = 1 + 1/log2(3).
    assert scores["ndcg"][0] == pytest.approx(1.5 / (1 + 1 / np.log2(3)))
    # Usuario 1: único relevante en la posición 2 → NDCG = (1/log2(3)) / 1; MRR = 1/2.
    assert scores["ndcg"][1] == pytest.approx(1 / np.log2(3))
    assert scores["mrr"].tolist() == pytest.approx([1.0, 0.5, 0.0])
    assert scores["precision"].tolist() == pytest.approx([2 / 3, 1 / 3, 0.0])
    assert scores["recall"].tolist() == pytest.approx([1.0, 1.0, 0.0])


def test_intra_list_diversity():
    # Ítems 0 y 1 idénticos, ítem 2 ortogonal a ambos.
    vectors = sp.csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))

    ild = intra_list_diversity(np.array([[0, 1, 2], [0, 1, 0]]), vectors, k=2)

    assert ild.tolist() == pytest.approx([0.0, 0.0])
    # Pares (0,1)=0, (0,2)=1, (1,2)=1 → ILD@3 = 2/3.
    assert intra_list_diversity(np.array([[0, 1, 2]]), vectors, k=3)[0] == pytest.approx(2 / 3)


def test_hybridization_rate():
    hybrid = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    als = np.array([[1, 2, 9, 10], [5, 6, 7, 8]])

    assert hybridization_rate(hybrid, als, k=4).tolist() == pytest.approx([0.5, 0.0])


def test_catalog_coverage():
    assert catalog_coverage(np.array([[0, 1], [1, 2]]), n_items=6) == pytest.approx(0.5)


def test_pre_post_table_marks_direction_of_improvement():
    table = pre_post_table(
        {"model": "popularity", "rmse": 1.2, "precision@10": 0.02},
        {"model": "hybrid", "rmse": 1.0, "precision@10": 0.05},
    ).set_index("metric")

    assert isinstance(table, pd.DataFrame)
    assert "model" not in table.index
    assert table.loc["rmse", "improved"]
    assert table.loc["precision@10", "improved"]
