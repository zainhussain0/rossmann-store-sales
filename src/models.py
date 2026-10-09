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
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data import training_frame
from src.features import FEATURES, StoreHistory


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


class GradientBoostingModel:
    """Gradient-boosted trees on log(Sales), one model for all stores.

    Uses scikit-learn's HistGradientBoostingRegressor, which is built on the
    same histogram-based algorithm as LightGBM (bin each feature, then grow
    trees on the bins) and handles NaN and categorical columns natively.

    Why one global model, not one per store: each store has only ~900 days of
    history, too little to learn holiday or promo effects reliably on its own.
    A global model learns those effects from all 1,115 stores at once, and
    the StoreHistory features tell it which store it's looking at.

    Why log(Sales): RMSPE scores *ratios* (see metrics.py). In log space a
    10% miss is the same distance for a small store and a big one, so plain
    squared error on log(Sales) is a close match for the competition metric.

    Expects `df` to already have the known-in-advance features from
    `features.prepare`. StoreHistory is fitted inside `fit`, so in
    cross-validation it only ever sees that fold's training period.
    """

    DEFAULT_PARAMS = {
        "learning_rate": 0.1,
        "max_iter": 500,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 100,  # leaves must cover ~100 store-days: resists one-off spikes
        "l2_regularization": 1.0,
        "categorical_features": "from_dtype",  # StoreType, Assortment, StateHoliday
        # sklearn would otherwise hold out a *random* 10% for early stopping,
        # which mixes future days into training. We validate by time instead.
        "early_stopping": False,
        "random_state": 42,
    }

    def __init__(self, params: dict | None = None, scale: float = 1.0):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        # Multiplier on final predictions. RMSPE rewards erring slightly low,
        # so a value a little under 1 may help; chosen later from validation.
        self.scale = scale

    def fit(self, df: pd.DataFrame) -> "GradientBoostingModel":
        self.history_ = StoreHistory().fit(df)
        train = self.history_.transform(training_frame(df))
        self.model_ = HistGradientBoostingRegressor(**self.params)
        self.model_.fit(train[FEATURES], np.log1p(train["Sales"]))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self.history_.transform(df)[FEATURES]
        pred = np.expm1(self.model_.predict(X)) * self.scale
        return np.where(df["Open"] == 1, pred, 0.0)
