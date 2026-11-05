"""Población y muestra: periodo 2022-01 a 2023-09 y usuarios con al menos 5 interacciones."""

from datetime import date

import pandas as pd
import pytest

from recsys.config import FilteringConfig, PeriodConfig
from recsys.data.preprocessing import (
    encode_ids,
    filter_active_users,
    filter_period,
    sample_report,
)
from recsys.schema import ITEM_ID, ITEM_IDX, TIMESTAMP, USER_ID, USER_IDX


def _ms(text: str) -> int:
    return int(pd.Timestamp(text, tz="UTC").timestamp() * 1000)


def test_active_users_filter_keeps_users_with_five_interactions(raw_reviews):
    filtered = filter_active_users(raw_reviews, min_interactions=5)

    assert filtered[USER_ID].value_counts().min() >= 5
    assert "u_inactive" not in set(filtered[USER_ID])


def test_active_users_filter_does_not_filter_items(raw_reviews):
    # i_rare tiene 1 interacción, pero pertenece a u0 (activo): la muestra no filtra ítems.
    filtered = filter_active_users(raw_reviews, min_interactions=5)

    assert "i_rare" in set(filtered[ITEM_ID])


def test_period_filter_includes_both_boundary_days():
    frame = pd.DataFrame(
        {
            USER_ID: ["a", "b", "c", "d"],
            ITEM_ID: ["x"] * 4,
            TIMESTAMP: [
                _ms("2021-12-31 23:59:59"),
                _ms("2022-01-01 00:00:00"),
                _ms("2023-09-30 23:59:59"),
                _ms("2023-10-01 00:00:00"),
            ],
        }
    )
    kept = filter_period(frame, date(2022, 1, 1), date(2023, 9, 30))

    assert list(kept[USER_ID]) == ["b", "c"]


def test_config_rejects_thresholds_below_thesis_minimum():
    with pytest.raises(ValueError):
        FilteringConfig(min_user_interactions=4)
    with pytest.raises(ValueError):
        PeriodConfig(start=date(2023, 9, 30), end=date(2022, 1, 1))


def test_sample_report_density(raw_reviews):
    report = sample_report(filter_active_users(raw_reviews, 5), n_reviews_in_period=100)

    assert report["n_users"] == 6
    assert report["n_items"] == 7
    assert report["density_pct"] == pytest.approx(round(100 * 37 / (6 * 7), 4))


def test_encode_ids_is_consecutive(raw_reviews):
    encoded, users, items = encode_ids(filter_active_users(raw_reviews, 5))

    assert sorted(encoded[USER_IDX].unique()) == list(range(len(users)))
    assert sorted(encoded[ITEM_IDX].unique()) == list(range(len(items)))
