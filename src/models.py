"""Forecasting models for Rossmann store sales.

Every model follows the same small interface, fit(train_df) -> self and
predict(df) -> array, so `validation.cross_validate` can score any of them
on identical folds.

Baselines come first. They answer "how good is a forecast with no machine
learning at all?", and every later model is judged by how much it beats them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import training_frame


class GroupMedianBaseline:
    """Predict the median historical sales for each group of `keys`.

    Why the median, not the mean: daily sales are right-skewed (EDA section 4)
    and contain one-off spikes. The median is a more typical day, and RMSPE
    punishes over-prediction harder than under-prediction (see metrics.py),
    so a value that leans low is also the safer choice.

    Fallbacks: a group in the forecast period may never appear in training
    (e.g. a store that starts opening on Sundays). Then we fall back to the
    store's overall median, then to the global median, rather than
    predicting NaN.
    """

    def __init__(self, keys: list[str]):
        self.keys = list(keys)

    def fit(self, df: pd.DataFrame) -> "GroupMedianBaseline":
        df = training_frame(df)  # open days with sales only (EDA section 2)
        self.group_median_ = df.groupby(self.keys)["Sales"].median().rename("pred")
        self.store_median_ = df.groupby("Store")["Sales"].median().rename("store_pred")
        self.global_median_ = float(df["Sales"].median())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        cols = list(dict.fromkeys(self.keys + ["Store", "Open"]))  # dedupe, keep order
        out = (
            df[cols]
            .join(self.group_median_, on=self.keys)
            .join(self.store_median_, on="Store")
        )
        pred = out["pred"].fillna(out["store_pred"]).fillna(self.global_median_)
        # Closed days are 0 by rule, not by model (EDA section 2).
        return np.where(out["Open"] == 1, pred, 0.0)


# The three baselines, simplest first. Each adds one effect the EDA found.
BASELINES = {
    "store": ["Store"],                                      # store size (EDA 4)
    "store_dow": ["Store", "DayOfWeek"],                     # + weekly pattern (EDA 7)
    "store_dow_promo": ["Store", "DayOfWeek", "Promo"],      # + promo lift (EDA 6)
}
