"""Prueba de hipótesis pareada (Shapiro-Wilk → t-Student relacionada o Wilcoxon)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from recsys.config import StatisticsConfig
from recsys.experiments.hypothesis_tests import hypothesis_table, paired_test

CONFIG = StatisticsConfig(alpha=0.05, alternative="greater", min_relative_improvement=0.10)


def test_normal_differences_use_paired_t_test():
    # Diferencias con forma normal exacta (cuantiles teóricos), sin depender del azar.
    o1 = np.random.default_rng(0).normal(0.10, 0.02, size=200)
    o2 = o1 + 0.03 + 0.01 * stats.norm.ppf((np.arange(200) + 0.5) / 200)

    result = paired_test(o1, o2, CONFIG)

    assert result.normal
    assert result.test.startswith("t-Student")
    assert result.significant
    assert result.meets_goal


def test_skewed_differences_use_wilcoxon():
    rng = np.random.default_rng(1)
    o1 = np.zeros(300)
    o2 = rng.exponential(0.05, size=300) * (rng.random(300) < 0.4)

    result = paired_test(o1, o2, CONFIG)

    assert not result.normal
    assert result.test.startswith("Wilcoxon")
    assert result.significant


def test_goal_requires_ten_percent_improvement():
    rng = np.random.default_rng(2)
    o1 = rng.normal(1.0, 0.05, size=500)
    o2 = o1 + rng.normal(0.02, 0.01, size=500)  # significativo pero solo ~2 %

    result = paired_test(o1, o2, CONFIG)

    assert result.significant
    assert result.relative_improvement == pytest.approx(0.02, abs=0.005)
    assert not result.meets_goal


def test_identical_models_do_not_reject_h0():
    values = np.full(50, 0.2)

    result = paired_test(values, values, CONFIG)

    assert not result.significant
    assert result.p_value == 1.0


def test_hypothesis_table_maps_indicators_to_hypotheses():
    users = np.arange(30)
    rng = np.random.default_rng(3)
    columns = ["precision@10", "recall@10", "ndcg@10", "mrr@10", "ild@10"]
    pre = pd.DataFrame({"user_idx": users} | {c: rng.random(30) * 0.1 for c in columns})
    post = pre.copy()
    post[columns] += 0.05

    table = hypothesis_table(pre, post, CONFIG, "user_idx").set_index("metric")

    assert table.loc["precision@10", "hypothesis"] == "HE1"
    assert table.loc["mrr@10", "indicator"] == "I10"
    assert table.loc["ild@10", "hypothesis"] == "HE3"
    assert set(table["k"]) == {10}


def test_hypothesis_table_rejects_different_users():
    pre = pd.DataFrame({"user_idx": [0, 1, 2], "precision@5": [0.1, 0.2, 0.3]})
    post = pd.DataFrame({"user_idx": [0, 1, 3], "precision@5": [0.1, 0.2, 0.3]})

    with pytest.raises(ValueError):
        hypothesis_table(pre, post, CONFIG, "user_idx")
