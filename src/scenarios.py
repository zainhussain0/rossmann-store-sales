"""What-if forecasts: "what happens to store X's sales if we change Y?"

A scenario edits the *controllable, known-in-advance* inputs for one store
over part of the forecast window (running a promo, opening or closing a day),
rebuilds that store's features, and compares the forecast to the unchanged
plan.

Read every result as "what the model expects", not "what would happen". The
model learned from historical associations: promos weren't randomly assigned
(EDA section 6), so a predicted uplift mixes the promo's real effect with
whatever else tended to come with promo days. It is a planning aid, not a
causal estimate.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.data import load_all
from src.features import add_known_features
from src.models import GradientBoostingModel

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "final_model.joblib"

# The only inputs a store manager could actually change. Holidays aren't
# levers, and everything else is derived from these by add_known_features.
LEVERS = {"Promo", "Open"}


def load_or_train(path: Path = ARTIFACT, retrain: bool = False) -> GradientBoostingModel:
    """The final model, fitted on *all* training data (2013-01 to 2015-07).

    Cross-validation chose the approach; the final model then uses every day
    available, as the real forecast would. Training takes about a minute, so
    the fitted model is saved and reused.
    """
    if path.exists() and not retrain:
        return joblib.load(path)
    train, _ = load_all()
    model = GradientBoostingModel().fit(add_known_features(train))
    path.parent.mkdir(exist_ok=True)
    joblib.dump(model, path)
    return model


def store_forecast(model, raw_train: pd.DataFrame, raw_test: pd.DataFrame, store: int,
                   changes: dict | None = None, start=None, end=None,
                   weekdays_only: bool = False) -> pd.DataFrame:
    """Daily forecast for one store over the test window, optionally edited.

    `changes` maps a lever to its new value, applied to days in [start, end]
    (only Monday to Friday if `weekdays_only`, matching how promos ran).
    The store's features are rebuilt *after* the edit: switching on a promo
    also changes DaysSincePromo / DaysUntilPromo on the surrounding days, and
    leaving those stale would feed the model an impossible combination.

    `raw_train` / `raw_test` are the merged frames from `data.load_all`, before
    `features.prepare`.
    """
    future = raw_test[raw_test["Store"] == store].copy()
    if future.empty:
        raise ValueError(f"Store {store} is not in the forecast window (test.csv covers 856 stores)")

    if changes:
        unknown = set(changes) - LEVERS
        if unknown:
            raise ValueError(f"Not a controllable lever: {sorted(unknown)}. Use {sorted(LEVERS)}")
        in_window = future["Date"].between(pd.Timestamp(start), pd.Timestamp(end))
        if weekdays_only:
            in_window &= future["DayOfWeek"] <= 5
        for col, value in changes.items():
            future.loc[in_window, col] = value
        _check_in_distribution(future)

    history = raw_train[raw_train["Store"] == store]
    full = add_known_features(pd.concat([history, future], ignore_index=True))
    fc = full[full["Sales"].isna()].reset_index(drop=True)
    fc["Forecast"] = model.predict(fc)
    return fc[["Date", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "Forecast"]]


def _check_in_distribution(future: pd.DataFrame) -> None:
    """Refuse scenarios the model has never seen anything like.

    In 2.5 years of data, promos ran Monday to Friday only (EDA section 6).
    A weekend promo would ask the trees to extrapolate, and they can't: they
    would silently answer with whatever the nearest weekday leaf says.
    """
    weekend_promo = (future["Promo"] == 1) & (future["DayOfWeek"] >= 6)
    if weekend_promo.any():
        raise ValueError("Promos never ran on weekends in the training data, so the "
                         "model can't forecast one. Restrict the promo to Mon-Fri.")
    # No check for promo on a closed day: it's common in the real data (Promo
    # is a chain-wide calendar flag that stays on when one store shuts), and
    # closed days are forecast as 0 regardless.


def compare(model, raw_train: pd.DataFrame, raw_test: pd.DataFrame, store: int,
            changes: dict, start, end, weekdays_only: bool = False) -> tuple[pd.DataFrame, dict]:
    """Planned vs scenario forecast, per day and in total.

    Totals cover the *whole* 48-day window, not just the edited days, because
    effects spill over: a promo can pull sales forward from the days after it.
    """
    plan = store_forecast(model, raw_train, raw_test, store)
    alt = store_forecast(model, raw_train, raw_test, store, changes, start, end, weekdays_only)
    daily = plan.rename(columns={"Forecast": "Planned"}).assign(
        Scenario=alt["Forecast"].values,
        ScenarioPromo=alt["Promo"].values,
        ScenarioOpen=alt["Open"].values,
    )
    daily["Difference"] = daily["Scenario"] - daily["Planned"]
    totals = {
        "planned": float(daily["Planned"].sum()),
        "scenario": float(daily["Scenario"].sum()),
        "difference": float(daily["Difference"].sum()),
        "pct_change": float(daily["Difference"].sum() / daily["Planned"].sum()),
    }
    return daily, totals
