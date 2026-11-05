"""R5: híbrido en dos etapas (candidatos por contenido, reordenados con ALS)."""

import numpy as np
import pandas as pd
import pytest

from recsys.config import ALSConfig, ContentConfig, HybridConfig
from recsys.data.dataset import InteractionData
from recsys.models.als import ALSRecommender
from recsys.models.content_based import ContentBasedRecommender
from recsys.models.hybrid import HybridRecommender, _min_max
from recsys.schema import ITEM_IDX, RATING, TIMESTAMP, USER_IDX

CONTENT = ContentConfig(max_features=100, ngram_range=(1, 1), min_df=1, n_candidates=3)
ALS = ALSConfig(factors=4, regularization=0.01, iterations=15, alpha=10.0)


@pytest.fixture
def items() -> pd.DataFrame:
    """Productos 0-4 de carreras y 5-9 de terror."""
    genre = ["racing"] * 5 + ["horror"] * 5
    return pd.DataFrame(
        {
            ITEM_IDX: range(10),
            "title": [f"{g} game{i}" for i, g in enumerate(genre)],
            "features": None,
            "description": None,
            "categories": [[g] for g in genre],
            "store": "shop",
        }
    )


@pytest.fixture
def train() -> InteractionData:
    """Usuarios 0-9 compran carreras y 10-19 terror; a cada uno le falta un producto."""
    rows = [
        (user, item, 5.0, user * 10 + item)
        for user in range(20)
        for item in (range(5) if user < 10 else range(5, 10))
        if item != (user % 5) + (0 if user < 10 else 5)
    ]
    return InteractionData(
        pd.DataFrame(rows, columns=[USER_IDX, ITEM_IDX, RATING, TIMESTAMP]), 20, 10
    )


def _hybrid(items, blend_weight: float) -> HybridRecommender:
    return HybridRecommender(
        ContentBasedRecommender(CONTENT, items),
        ALSRecommender(ALS, seed=0),
        HybridConfig(blend_weight=blend_weight),
        n_candidates=CONTENT.n_candidates,
    )


def test_only_content_candidates_get_a_score(items, train):
    model = _hybrid(items, blend_weight=0.7).fit(train)
    scores = model._score_items(np.array([0, 13]))

    assert (np.isfinite(scores).sum(axis=1) == CONTENT.n_candidates).all()
    assert scores[np.isfinite(scores)].min() >= 0 and scores.max() <= 1
    assert model.recommend(np.array([0]), k=1)[0].tolist() == [0]
    assert model.recommend(np.array([13]), k=1)[0].tolist() == [8]


def test_blend_weight_one_ranks_candidates_by_als(items, train):
    model = _hybrid(items, blend_weight=1.0).fit(train)
    users = np.array([0])
    candidates = np.flatnonzero(np.isfinite(model._score_items(users)[0]))

    als_order = candidates[np.argsort(-model.als._score_items(users)[0, candidates])]
    assert model.recommend(users, k=3)[0].tolist() == als_order.tolist()


def test_training_time_includes_both_stages(items, train):
    model = _hybrid(items, blend_weight=0.5).fit(train)

    assert model.train_time_s >= model.content.train_time_s + model.als.train_time_s


def test_predicted_ratings_blend_both_stages(items, train):
    model = _hybrid(items, blend_weight=0.5).fit(train)
    users, targets = np.array([0, 13]), np.array([0, 8])

    expected = (
        model.als.predict_rating(users, targets) + model.content.predict_rating(users, targets)
    ) / 2
    np.testing.assert_allclose(model.predict_rating(users, targets), expected)


def test_min_max_scales_rows_and_handles_constant_rows():
    scaled = _min_max(np.array([[1.0, 3.0, 2.0], [4.0, 4.0, 4.0]]))

    np.testing.assert_allclose(scaled, [[0.0, 1.0, 0.5], [0.0, 0.0, 0.0]])
