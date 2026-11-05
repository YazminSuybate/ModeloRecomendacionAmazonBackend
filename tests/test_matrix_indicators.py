"""Indicadores de la matriz sobre los modelos entrenados: I1, I2, I6, I7-I12 y tiempo de ALS."""

import numpy as np
import pandas as pd
import pytest

from recsys.config import ALSConfig, ContentConfig, EvaluationConfig, HybridConfig
from recsys.data.dataset import InteractionData
from recsys.evaluation.evaluator import evaluate_with_details
from recsys.evaluation.hybridization import hybridization_rate
from recsys.models.als import ALSRecommender
from recsys.models.content_based import ContentBasedRecommender
from recsys.models.hybrid import HybridRecommender
from recsys.schema import ITEM_IDX, RATING, TIMESTAMP, USER_IDX

CONTENT = ContentConfig(
    max_features=100, ngram_range=(1, 1), min_df=1, n_candidates=4, text_fields=("title",)
)
ALS = ALSConfig(factors=4, regularization=0.01, iterations=10, alpha=10.0)


@pytest.fixture
def items() -> pd.DataFrame:
    genre = ["racing"] * 5 + ["horror"] * 5
    titles = [f"{g} game{i}" for i, g in enumerate(genre)]
    return pd.DataFrame({ITEM_IDX: range(10), "title": titles})


@pytest.fixture
def data() -> tuple[InteractionData, InteractionData]:
    """Train: cada usuario compra 3 de su género. Test: el 4.º producto de su género."""
    train, test = [], []
    for user in range(12):
        base = 0 if user < 6 else 5
        owned = [base + (user + j) % 5 for j in range(4)]
        train += [(user, item, 5.0, t) for t, item in enumerate(owned[:3])]
        test.append((user, owned[3], 5.0, 10))
    columns = [USER_IDX, ITEM_IDX, RATING, TIMESTAMP]
    return (
        InteractionData(pd.DataFrame(train, columns=columns), 12, 10),
        InteractionData(pd.DataFrame(test, columns=columns), 12, 10),
    )


@pytest.fixture
def hybrid(items, data) -> HybridRecommender:
    return HybridRecommender(
        ContentBasedRecommender(CONTENT, items),
        ALSRecommender(ALS, seed=0),
        HybridConfig(blend_weight=0.7),
        CONTENT.n_candidates,
    ).fit(data[0])


def test_content_reports_i1_and_i2(hybrid):
    assert hybrid.content.n_attributes == 1
    assert hybrid.content.vector_dimension == len(hybrid.content.vectorizer.vocabulary_)
    assert hybrid.content.item_vectors.shape[0] == 10


def test_als_factorization_time_excludes_calibration(hybrid):
    assert 0 < hybrid.als.factorization_time_s <= hybrid.als.train_time_s


def test_stage_scores_match_the_final_ranking(hybrid):
    users = np.arange(12)
    candidates, _, _, final = hybrid.stage_scores(users)
    order = np.argsort(-final, axis=1, kind="stable")[:, :3]

    np.testing.assert_array_equal(
        np.take_along_axis(candidates, order, axis=1), hybrid.recommend(users, k=3)
    )


def test_evaluation_reports_every_vd_indicator_per_user(hybrid, data):
    config = EvaluationConfig(k_values=(2, 3), relevance_threshold=4.0, batch_size=5)

    result = evaluate_with_details(hybrid, data[1], config, hybrid.content.item_vectors)

    for k in (2, 3):
        for metric in ("precision", "recall", "ndcg", "mrr", "ild"):
            assert f"{metric}@{k}" in result.per_user
            assert result.summary[f"{metric}@{k}"] == pytest.approx(
                result.per_user[f"{metric}@{k}"].mean()
            )
        assert f"coverage@{k}" in result.summary
    assert len(result.per_user) == 12
    assert result.recommended.shape == (12, 3)


def test_hybridization_rate_against_its_own_als(hybrid, data):
    config = EvaluationConfig(k_values=(3,), relevance_threshold=4.0, batch_size=5)
    post = evaluate_with_details(hybrid, data[1], config)
    als = evaluate_with_details(hybrid.als, data[1], config)

    rate = hybridization_rate(post.recommended, als.recommended, k=3)

    assert ((rate >= 0) & (rate <= 1)).all()
