import numpy as np
import pandas as pd

from src.data import load_all
from src.features import prepare
from src.models import GroupMedianBaseline, GradientBoostingModel
from src.validation import make_folds, split


def _train():
    # Store 1: Mondays sell 100/300 (median 200), Tuesdays 50.
    # A closed day and a zero-sales open day must be ignored by fit.
    return pd.DataFrame({
        "Store":     [1,   1,   1,  1, 1],
        "DayOfWeek": [1,   1,   2,  2, 2],
        "Open":      [1,   1,   1,  0, 1],
        "Sales":     [100, 300, 50, 0, 0],
    })


def test_group_median_and_closed_rule():
    m = GroupMedianBaseline(["Store", "DayOfWeek"]).fit(_train())
    test = pd.DataFrame({"Store": [1, 1, 1], "DayOfWeek": [1, 2, 1], "Open": [1, 1, 0]})
    np.testing.assert_array_equal(m.predict(test), [200, 50, 0])


def test_unseen_group_falls_back_to_store_then_global():
    store2 = pd.DataFrame({"Store": [2, 2], "DayOfWeek": [1, 1], "Open": [1, 1], "Sales": [1000, 1000]})
    m = GroupMedianBaseline(["Store", "DayOfWeek"]).fit(pd.concat([_train(), store2]))
    # Store 1 never seen on Sunday -> store 1 median (100).
    # Store 3 never seen at all -> global median of [100, 300, 50, 1000, 1000] = 300.
    test = pd.DataFrame({"Store": [1, 3], "DayOfWeek": [7, 1], "Open": [1, 1]})
    np.testing.assert_array_equal(m.predict(test), [100, 300])


def test_single_key_works():
    m = GroupMedianBaseline(["Store"]).fit(_train())
    test = pd.DataFrame({"Store": [1], "Open": [1]})
    np.testing.assert_array_equal(m.predict(test), [100])


def test_gradient_boosting_smoke_on_real_subset():
    # Small and fast: 20 stores, 20 trees. Checks wiring, not accuracy.
    train, test = prepare(*load_all())
    train = train[train["Store"] <= 20]
    fit_df, val = split(train, make_folds(train["Date"].max(), n_folds=1)[0])
    model = GradientBoostingModel({"max_iter": 20}).fit(fit_df)
    pred = model.predict(val)
    assert len(pred) == len(val) and np.isfinite(pred).all()
    assert (pred[val["Open"] == 0] == 0).all()
    assert (pred[val["Open"] == 1] > 0).all()
