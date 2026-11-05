"""R4: Etapa 2 con ALS, y R7: calibración de sus puntajes a ratings 1-5."""

import numpy as np
import pandas as pd
import pytest

from recsys.config import ALSConfig
from recsys.data.dataset import InteractionData
from recsys.models.als import ALSRecommender
from recsys.models.calibration import RatingCalibrator
from recsys.schema import ITEM_IDX, RATING, TIMESTAMP, USER_IDX

CONFIG = ALSConfig(factors=4, regularization=0.01, iterations=15, alpha=10.0)


@pytest.fixture
def two_communities() -> InteractionData:
    """Usuarios 0-9 compran productos 0-4 y usuarios 10-19 compran productos 5-9.

    Cada usuario deja sin comprar un producto de su comunidad.
    """
    rows = [
        (user, item, 5.0 if item % 2 else 4.0, user * 10 + item)
        for user in range(20)
        for item in (range(5) if user < 10 else range(5, 10))
        if item != (user % 5) + (0 if user < 10 else 5)
    ]
    frame = pd.DataFrame(rows, columns=[USER_IDX, ITEM_IDX, RATING, TIMESTAMP])
    return InteractionData(frame, n_users=20, n_items=10)


def test_confidence_combines_existence_and_rating(two_communities):
    model = ALSRecommender(CONFIG, seed=0)
    confidence = model.confidence(two_communities)

    assert confidence[0, 1] == pytest.approx(1 + 10 * 5 / 5)  # rating 5
    assert confidence[0, 2] == pytest.approx(1 + 10 * 4 / 5)  # rating 4
    assert confidence.nnz == len(two_communities.frame)  # sin interacción no hay entrada


def test_recommends_the_missing_item_of_the_community(two_communities):
    model = ALSRecommender(CONFIG, seed=0).fit(two_communities)

    assert model.recommend(np.array([0]), k=1)[0].tolist() == [0]
    assert model.recommend(np.array([13]), k=1)[0].tolist() == [8]
    assert model.train_time_s >= 0


def test_predicted_ratings_are_on_the_rating_scale(two_communities):
    model = ALSRecommender(CONFIG, seed=0).fit(two_communities)
    predictions = model.predict_rating(np.arange(20), np.arange(20) % 10)

    assert ((predictions >= 1) & (predictions <= 5)).all()


def test_same_seed_gives_same_model(two_communities):
    first = ALSRecommender(CONFIG, seed=7).fit(two_communities)
    second = ALSRecommender(CONFIG, seed=7).fit(two_communities)
    users = np.arange(20)

    np.testing.assert_allclose(first._score_items(users), second._score_items(users), rtol=1e-5)


def test_calibrator_learns_a_linear_map_and_clips_to_the_scale():
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    calibrator = RatingCalibrator().fit(features, np.array([1.0, 2.0, 3.0, 4.0]))

    assert calibrator.predict(np.array([[1.5]])) == pytest.approx([2.5])
    assert calibrator.predict(np.array([[-10.0], [10.0]])).tolist() == [1.0, 5.0]
