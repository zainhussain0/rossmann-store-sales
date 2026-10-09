import pandas as pd

from src.validation import HORIZON, make_folds, seasonal_fold, split

LAST = pd.Timestamp("2015-07-31")


def test_default_fold_dates():
    folds = make_folds(LAST)
    assert [(f.val_start, f.val_end) for f in folds] == [
        (pd.Timestamp("2015-03-10"), pd.Timestamp("2015-04-26")),
        (pd.Timestamp("2015-04-27"), pd.Timestamp("2015-06-13")),
        (pd.Timestamp("2015-06-14"), pd.Timestamp("2015-07-31")),
    ]


def test_each_fold_is_horizon_days_and_trains_on_the_day_before():
    for f in make_folds(LAST) + [seasonal_fold()]:
        assert (f.val_end - f.val_start).days + 1 == HORIZON
        assert f.train_end == f.val_start - pd.Timedelta(days=1)


def test_seasonal_fold_matches_test_calendar_window():
    f = seasonal_fold()
    assert (f.val_start, f.val_end) == (pd.Timestamp("2014-08-01"), pd.Timestamp("2014-09-17"))


def test_split_has_no_future_leakage():
    df = pd.DataFrame({"Date": pd.date_range("2015-01-01", LAST)})
    for f in make_folds(LAST):
        train, val = split(df, f)
        assert train["Date"].max() == f.train_end
        assert val["Date"].min() == f.val_start
        assert val["Date"].max() == f.val_end
        assert len(val) == HORIZON
