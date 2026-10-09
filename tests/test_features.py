import numpy as np
import pandas as pd

from src.features import (EVENT_DAYS_CAP, FEATURES, StoreHistory, _add_competition,
                          _add_promo2, _days_to_event, add_known_features)
from src.data import load_all


def test_competition_months_open():
    df = pd.DataFrame({
        "Year": [2015] * 4, "Month": [3] * 4,
        "CompetitionOpenSinceYear":  [2014, 2015, 1900, np.nan],
        "CompetitionOpenSinceMonth": [12,   6,    1,    np.nan],
    })
    out = _add_competition(df)["CompetitionMonthsOpen"]
    # 3 months open; not open yet; absurd 1900 capped; unknown stays NaN.
    assert out.tolist()[:3] == [3, -1, 24]
    assert np.isnan(out.iloc[3])


def test_promo2_needs_started_and_right_month():
    dates = pd.to_datetime(["2014-09-10", "2015-09-10", "2015-10-10", "2015-09-10"])
    df = pd.DataFrame({
        "Store": [1, 1, 1, 2], "Date": dates, "Month": dates.month,
        "Promo2": [1, 1, 1, 0],
        "Promo2SinceYear": [2015, 2015, 2015, 0], "Promo2SinceWeek": [1, 1, 1, 0],
        "PromoInterval": ["Mar,Jun,Sept,Dec"] * 3 + ["None"],
    })
    out = _add_promo2(df)
    # Before campaign start; active (and 'Sept' parsed); wrong month; never joined.
    assert out["Promo2Active"].tolist() == [0, 1, 0, 0]
    assert np.isnan(out["Promo2Weeks"].iloc[3])


def test_days_since_and_until_event_per_store():
    df = pd.DataFrame({
        "Store": [1, 1, 1, 1, 2, 2],
        "Date": pd.to_datetime(["2015-01-01", "2015-01-02", "2015-01-03", "2015-01-04",
                                "2015-01-01", "2015-01-02"]),
    })
    flag = pd.Series([False, True, False, False, False, False])
    assert _days_to_event(df, flag, "since").tolist() == [EVENT_DAYS_CAP, 0, 1, 2, EVENT_DAYS_CAP, EVENT_DAYS_CAP]
    assert _days_to_event(df, flag, "until").tolist() == [1, 0, EVENT_DAYS_CAP, EVENT_DAYS_CAP, EVENT_DAYS_CAP, EVENT_DAYS_CAP]


def test_store_history_uses_open_days_and_leaves_unseen_stores_nan():
    train = pd.DataFrame({
        "Store": [1, 1, 1], "DayOfWeek": [1, 2, 3], "Promo": [0, 1, 0],
        "Open": [1, 1, 0], "Sales": [np.e - 1, np.e ** 3 - 1, 0], "Customers": [1, 1, 0],
        "Date": pd.to_datetime(["2015-01-05", "2015-01-06", "2015-01-07"]),
    })
    hist = StoreHistory().fit(train)
    out = hist.transform(pd.DataFrame({"Store": [1, 2], "DayOfWeek": [1, 1], "Promo": [1, 1]}))
    # log1p(sales) on the two open days is 1 and 3, so the store mean is 2; the closed day is ignored.
    assert out["StoreMeanLogSales"].iloc[0] == 2.0
    assert out["StorePromoMeanLogSales"].iloc[0] == 3.0
    assert out.iloc[1][["StoreMeanLogSales", "StoreDowMeanLogSales"]].isna().all()


def test_full_pipeline_on_real_data():
    train, test = load_all()
    full = add_known_features(pd.concat([train, test], ignore_index=True))
    assert len(full) == len(train) + len(test)
    full = StoreHistory().fit(full[full["Sales"].notna()]).transform(full)
    missing = set(FEATURES) - set(full.columns)
    assert not missing, missing
    assert "Customers" not in FEATURES and "Sales" not in FEATURES
