"""Time-based validation splits for the Rossmann forecast.

Design rule: validation must copy the real forecasting setup. In test we know
everything up to 2015-07-31 and forecast the next 48 days (EDA section 1), so
every fold trains on the past and validates on the 48 days straight after it.

Random k-fold would be wrong here: it puts days from the future into the
training set, so the model gets to "see" next month while predicting this one.
Scores look great and then fall apart on the real test window.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.metrics import rmspe

HORIZON = 48  # days in test.csv: 2015-08-01 to 2015-09-17


@dataclass(frozen=True)
class Fold:
    """One train/validation split. All dates are inclusive."""
    name: str
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def make_folds(last_date, n_folds: int = 3, horizon: int = HORIZON) -> list[Fold]:
    """Back-to-back validation windows ending at `last_date`, oldest first.

    Each fold is "expanding window": it trains on *all* history before its
    validation window, not a fixed-length slice. That matches test, where
    we'll train on everything we have.

    Why several folds instead of one: a single 48-day window can be unusual
    (Easter 2015 falls in the earliest default fold). Averaging over three
    windows tells us whether a change helps in general or got lucky once.

    With the default arguments and train ending 2015-07-31, the validation
    windows are 2015-03-10..04-26, 04-27..06-13 and 06-14..07-31.
    """
    last_date = pd.Timestamp(last_date)
    folds = []
    for i in range(n_folds, 0, -1):
        val_end = last_date - pd.Timedelta(days=horizon * (i - 1))
        val_start = val_end - pd.Timedelta(days=horizon - 1)
        folds.append(Fold(
            name=f"fold_{val_start:%Y-%m-%d}",
            train_end=val_start - pd.Timedelta(days=1),
            val_start=val_start,
            val_end=val_end,
        ))
    return folds


def seasonal_fold(horizon: int = HORIZON) -> Fold:
    """A fold covering the same calendar window as test, one year earlier.

    The default folds are all spring/summer 2015, but test is Aug-Sep. This
    fold checks the model on the right time of year.

    Caveat: 180 stores have no data at all from 2014-07-01 to 2014-12-31, so
    this fold scores only the other 935 stores, and its training data ends
    in June 2014 for the missing 180. Use it as a secondary check alongside
    `make_folds`, never as the only score.
    """
    val_start = pd.Timestamp("2014-08-01")
    return Fold(
        name="seasonal_2014-08-01",
        train_end=val_start - pd.Timedelta(days=1),
        val_start=val_start,
        val_end=val_start + pd.Timedelta(days=horizon - 1),
    )


def split(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, val) rows of `df` for one fold, split on `Date`.

    Days after `val_end` are dropped entirely: for an earlier fold, they are
    the future and must not be used for training or validation.
    """
    train = df[df["Date"] <= fold.train_end]
    val = df[(df["Date"] >= fold.val_start) & (df["Date"] <= fold.val_end)]
    # Cheap guard against a leak if the logic above is ever edited.
    assert train["Date"].max() < val["Date"].min(), f"{fold.name}: train overlaps val"
    return train.reset_index(drop=True), val.reset_index(drop=True)


def cross_validate(make_model, df: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    """Fit a fresh model per fold and score it with RMSPE.

    `make_model` is a zero-argument function returning an unfitted model, so
    no fitted state can carry over from one fold into the next.
    """
    rows = []
    for fold in folds:
        train, val = split(df, fold)
        pred = make_model().fit(train).predict(val)
        rows.append({"fold": fold.name, "rmspe": rmspe(val["Sales"], pred)})
    return pd.DataFrame(rows)
