from recsys.data.dataset import InteractionData
from recsys.data.preprocessing import encode_ids, filter_active_users
from recsys.data.splitting import temporal_holdout, temporal_user_split
from recsys.data.storage import train_with_validation
from recsys.schema import TIMESTAMP, USER_IDX


def test_split_is_temporal_and_covers_every_user(raw_reviews):
    encoded, _, _ = encode_ids(filter_active_users(raw_reviews, 5))
    train, val, test = temporal_user_split(encoded, val_ratio=0.2, test_ratio=0.2)

    assert len(train) + len(val) + len(test) == len(encoded)
    for split in (train, val, test):
        assert set(split[USER_IDX]) == set(encoded[USER_IDX])

    last_train = train.groupby(USER_IDX)[TIMESTAMP].max()
    first_val = val.groupby(USER_IDX)[TIMESTAMP].min()
    last_val = val.groupby(USER_IDX)[TIMESTAMP].max()
    first_test = test.groupby(USER_IDX)[TIMESTAMP].min()
    assert (last_train < first_val).all()
    assert (last_val < first_test).all()


def test_test_is_the_same_20_percent_as_the_thesis_holdout(raw_reviews):
    # La validación sale del 80 % de train: test es idéntico a un holdout 80/20 directo.
    encoded, _, _ = encode_ids(filter_active_users(raw_reviews, 5))
    train, val, test = temporal_user_split(encoded, val_ratio=0.2, test_ratio=0.2)
    train_80, test_20 = temporal_holdout(encoded, 0.2)

    assert test.equals(test_20)
    assert len(train) + len(val) == len(train_80)


def test_train_with_validation_joins_both_splits_and_keeps_shape(train_data):
    val = InteractionData(train_data.frame.iloc[:2], train_data.n_users, train_data.n_items)
    splits = {"train": train_data, "val": val, "test": val}

    joined = train_with_validation(splits)

    assert len(joined.frame) == len(train_data.frame) + 2
    assert (joined.n_users, joined.n_items) == (train_data.n_users, train_data.n_items)
