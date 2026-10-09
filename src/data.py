"""Data loading and cleaning for the Rossmann store-sales forecast.

Design rule: this module only loads and cleans. Anything that creates
predictive signal (lags, rolling means, competition age) lives in features.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# PromoInterval uses "Sept" for September, but pandas/strftime use "Sep".
# Matching month names by string would silently drop every September store.
MONTH_LOOKUP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Sept": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def parse_promo_months(interval) -> tuple:
    """'Jan,Apr,Jul,Oct' -> (1, 4, 7, 10). Missing / 'None' -> ()."""
    if interval is None or pd.isna(interval) or interval == "None":
        return ()
    return tuple(MONTH_LOOKUP[m.strip()] for m in interval.split(","))


def load_store(path=None) -> pd.DataFrame:
    """Load store.csv and clean it. One row per store."""
    store = pd.read_csv(path or DATA_DIR / "store.csv")

    # 3 stores have no competition distance: genuinely unknown, so use the
    # median (robust to the long right tail) and keep a flag so a model can
    # learn that these values were imputed.
    store["CompetitionDistanceMissing"] = store["CompetitionDistance"].isna().astype(int)
    store["CompetitionDistance"] = store["CompetitionDistance"].fillna(
        store["CompetitionDistance"].median()
    )

    # Promo2 fields are *structurally* missing for stores with Promo2 == 0:
    # the value doesn't exist, it isn't unknown. Use sentinels, not imputation.
    store["Promo2SinceWeek"] = store["Promo2SinceWeek"].fillna(0)
    store["Promo2SinceYear"] = store["Promo2SinceYear"].fillna(0)
    store["PromoInterval"] = store["PromoInterval"].fillna("None")

    # CompetitionOpenSince* are left as NaN on purpose: whether a competitor
    # exists *yet* depends on the row's date, so features.py handles it.
    return store


def load_train(path=None) -> pd.DataFrame:
    df = pd.read_csv(
        path or DATA_DIR / "train.csv",
        parse_dates=["Date"],
        dtype={"StateHoliday": str},  # mixed 0 / 'a' / 'b' / 'c' in the raw file
        low_memory=False,
    )
    return df.sort_values(["Store", "Date"]).reset_index(drop=True)


def load_test(path=None) -> pd.DataFrame:
    df = pd.read_csv(
        path or DATA_DIR / "test.csv",
        parse_dates=["Date"],
        dtype={"StateHoliday": str},
    )
    # 11 missing Open values, all store 622 on Mon-Sat. In training that store
    # is open Mon-Sat and shut on Sundays, so "open" is the evidence-backed fill.
    df["Open"] = df["Open"].fillna(1).astype(int)
    return df


def merge_store(df: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """Left-join store attributes; validate that every row finds its store."""
    out = df.merge(store, on="Store", how="left", validate="many_to_one")
    assert out["StoreType"].notna().all(), "Some rows have no matching store"
    return out


def training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows the model should learn from: open days with real sales.

    Closed days are always zero sales, so they teach the model 'closed => 0'
    rather than anything about demand. At prediction time we set closed days
    to 0 directly and only forecast the open ones.
    """
    return df[(df["Open"] == 1) & (df["Sales"] > 0)].reset_index(drop=True)


def load_all():
    """Convenience: (train, test) each merged with the store table."""
    store = load_store()
    return merge_store(load_train(), store), merge_store(load_test(), store)