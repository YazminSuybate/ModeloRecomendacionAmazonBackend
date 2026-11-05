"""R3: Etapa 1, filtrado basado en contenido."""

import numpy as np
import pandas as pd
import pytest

from recsys.config import ContentConfig
from recsys.data.dataset import InteractionData
from recsys.models.content_based import ContentBasedRecommender
from recsys.schema import ITEM_IDX, RATING, TIMESTAMP, USER_IDX

CONFIG = ContentConfig(max_features=100, ngram_range=(1, 1), min_df=1, n_candidates=2)


@pytest.fixture
def items() -> pd.DataFrame:
    """Dos productos de carreras y dos de terror; al último le faltan campos (se imputan)."""
    return pd.DataFrame(
        {
            ITEM_IDX: [0, 1, 2, 3],
            "title": ["racing cars", "racing bikes", "horror zombies", "horror ghosts"],
            "features": [["fast racing"], ["racing"], ["zombies"], []],
            "description": [["cars"], [], ["survival horror"], None],
            "categories": [["Racing"], ["Racing"], ["Horror"], ["Horror"]],
            "store": ["A", "B", "C", "D"],
        }
    )


@pytest.fixture
def train() -> InteractionData:
    frame = pd.DataFrame(
        {
            USER_IDX: [0, 1, 1],
            ITEM_IDX: [0, 2, 0],
            RATING: [5.0, 5.0, 1.0],
            TIMESTAMP: [1, 2, 3],
        }
    )
    return InteractionData(frame, n_users=2, n_items=4)


def test_recommends_items_similar_to_the_profile(items, train):
    model = ContentBasedRecommender(CONFIG, items).fit(train)

    assert model.recommend(np.array([0]), k=1)[0].tolist() == [1]  # racing → racing
    assert model.recommend(np.array([1]), k=1)[0].tolist() == [3]  # el 5 en horror pesa más


def test_scores_are_cosine_similarities(items, train):
    model = ContentBasedRecommender(CONFIG, items).fit(train)
    scores = model._score_items(np.array([0, 1]))

    assert scores.shape == (2, 4)
    assert scores.min() >= 0
    assert scores.max() == pytest.approx(1.0)  # perfil del usuario 0 = vector del producto 0


def test_predicted_rating_follows_similar_items(items, train):
    model = ContentBasedRecommender(CONFIG, items).fit(train)
    predictions = model.predict_rating(np.array([1, 1]), np.array([3, 1]))

    # Horror se parece al producto que calificó con 5; racing al que calificó con 1.
    assert predictions[0] > 4 and predictions[1] < 2
    assert ((predictions >= 1) & (predictions <= 5)).all()


def test_predicted_rating_falls_back_to_user_mean(items):
    items = items.assign(title=["alpha", "beta", "gamma", "delta"], features=None, description=None)
    items = items.assign(categories=None, store=["north", "south", "east", "west"])
    train = InteractionData(
        pd.DataFrame({USER_IDX: [0, 0], ITEM_IDX: [0, 1], RATING: [2.0, 4.0], TIMESTAMP: [1, 2]}),
        n_users=1,
        n_items=4,
    )
    model = ContentBasedRecommender(CONFIG, items).fit(train)

    # Sin palabras en común con el historial, se predice la media del usuario.
    assert model.predict_rating(np.array([0]), np.array([2])) == pytest.approx([3.0])
