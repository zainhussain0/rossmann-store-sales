"""Feature engineering for the Rossmann forecast.

The central constraint: we forecast 48 days ahead (EDA section 1). On the last
day of the test window, the most recent sales we know are 48 days old. So the
classic "sales yesterday" / "sales last week" lags are not available, and
building them would be a leak that validation can't fully catch.

Features therefore come in two kinds:

1. Known in advance (`add_known_features`): calendar, promo and holiday
   flags, store attributes. test.csv gives these for every forecast day, so
   they can be computed row by row with no risk.

2. Learned from history (`StoreHistory`): each store's typical sales level,
   weekly shape, promo response, etc. These use the target, so they must be
   *fitted on the training period only* and then looked up for the forecast
   period, exactly like a model. Fitting them on all data would leak the
   validation period's sales into the features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import parse_promo_months, training_frame

# Beyond these limits the exact value stops mattering: a competitor open for
# 2 years or 20 years is just "established", and a holiday 30 days away has no
# effect on today. Capping also hides the two absurd competition years in
# store.csv (1900 for store 146, 1961 for store 815).
COMPETITION_MONTHS_CAP = 24
EVENT_DAYS_CAP = 15

# Events whose *proximity* moves sales: shoppers stock up before a holiday or
# closure and catch up after it. All are known in advance via test.csv.
EVENTS = {
    "StateHoliday": lambda df: df["StateHoliday"] != "0",
    "SchoolHoliday": lambda df: df["SchoolHoliday"] == 1,
    "Promo": lambda df: df["Promo"] == 1,
    "Closed": lambda df: df["Open"] == 0,
}

# Fixed category lists, so a value always gets the same internal code no
# matter which rows are being processed. Inferring them from the data would
# give a single store (one StoreType) or test (no StateHoliday 'b'/'c')
# different codes from the ones the model was trained on.
CATEGORIES = {
    "StoreType": ["a", "b", "c", "d"],
    "Assortment": ["a", "b", "c"],
    "StateHoliday": ["0", "a", "b", "c"],
}


def add_known_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features that are known before the forecast day.

    Call this on train and test *concatenated*, sorted by Store and Date, so
    that "days until next holiday" on the last training days can see holidays
    in the test period. That isn't a leak: holiday and promo calendars are
    published in advance, which is exactly why they appear in test.csv.
    """
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    df = _add_calendar(df)
    df = _add_competition(df)
    df = _add_promo2(df)
    for name, is_event in EVENTS.items():
        flag = is_event(df)
        df[f"DaysSince{name}"] = _days_to_event(df, flag, direction="since")
        df[f"DaysUntil{name}"] = _days_to_event(df, flag, direction="until")
    for col, cats in CATEGORIES.items():
        df[col] = pd.Categorical(df[col], categories=cats)
    return df


def prepare(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add known features to train and test together, then split them again.

    Combined, so event distances on the last training days can see the
    test period's holidays and promos (see `add_known_features`).
    """
    full = add_known_features(pd.concat([train, test], ignore_index=True))
    is_train = full["Sales"].notna()
    return full[is_train].reset_index(drop=True), full[~is_train].reset_index(drop=True)


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = df["Date"].dt
    df["Year"] = d.year
    df["Month"] = d.month
    df["DayOfMonth"] = d.day
    df["WeekOfYear"] = d.isocalendar().week.astype(int)
    df["DayOfYear"] = d.dayofyear
    return df


def _add_competition(df: pd.DataFrame) -> pd.DataFrame:
    """Months the nearest competitor has been open, as of each row's date.

    store.csv gives a fixed opening month; whether the competitor exists
    *yet* depends on the row's date (why data.py left these NaN).
    -1 = competitor not open yet; NaN = opening date unknown (354 stores).
    """
    months = (
        (df["Year"] - df["CompetitionOpenSinceYear"]) * 12
        + (df["Month"] - df["CompetitionOpenSinceMonth"])
    )
    # mask(<0), not where(>=0): NaN >= 0 is False, so where() would turn
    # "unknown" into "not open yet".
    df["CompetitionMonthsOpen"] = months.mask(months < 0, -1).clip(upper=COMPETITION_MONTHS_CAP)
    return df


def _add_promo2(df: pd.DataFrame) -> pd.DataFrame:
    """Is the store's recurring Promo2 campaign running in this row's month?

    Promo2 has three conditions: the store takes part, the campaign has
    started (an ISO year/week in store.csv), and the month is one of the
    restart months in PromoInterval. The raw `Promo2` flag only captures the
    first, so on its own it says "this store sometimes runs Promo2".
    """
    stores = df.drop_duplicates("Store").set_index("Store")
    joined = stores["Promo2"] == 1
    start = pd.Series(pd.NaT, index=stores.index)
    start[joined] = pd.to_datetime(
        stores.loc[joined, "Promo2SinceYear"].astype(int).astype(str)
        + "-W" + stores.loc[joined, "Promo2SinceWeek"].astype(int).astype(str).str.zfill(2)
        + "-1",
        format="%G-W%V-%u",
    )
    months = stores["PromoInterval"].map(parse_promo_months)

    row_start = df["Store"].map(start)
    started = df["Date"] >= row_start  # NaT compares False -> non-participants
    in_month = [m in ms for m, ms in zip(df["Month"], df["Store"].map(months))]
    df["Promo2Active"] = (started & np.array(in_month)).astype(int)
    df["Promo2Weeks"] = ((df["Date"] - row_start).dt.days // 7).clip(lower=0)  # NaN if never
    return df


def _days_to_event(df: pd.DataFrame, flag: pd.Series, direction: str) -> pd.Series:
    """Days since the last / until the next event for the same store.

    Assumes df is sorted by Store, Date. Capped at EVENT_DAYS_CAP; "no event
    in range" (including off the edge of the data) also gets the cap.
    """
    event_date = df["Date"].where(flag)
    by_store = event_date.groupby(df["Store"])
    if direction == "since":
        days = (df["Date"] - by_store.ffill()).dt.days
    else:
        days = (by_store.bfill() - df["Date"]).dt.days
    return days.fillna(EVENT_DAYS_CAP).clip(upper=EVENT_DAYS_CAP).astype(int)


class StoreHistory:
    """Store-level summaries learned from the training period only.

    These replace short lags. Instead of "sales 7 days ago" (unknown 48 days
    out), the model gets "this store's typical log sales on a Tuesday with a
    promo", which is equally valid on day 1 and day 48 of the forecast.

    Note on Customers: it can't be a same-day feature (absent from test, EDA
    section 3), but its *historical average* is fine: by the forecast date,
    past customer counts are known facts.
    """

    RECENT_DAYS = 90  # "recent level" window, to pick up trends since the start of the data

    def fit(self, train: pd.DataFrame) -> "StoreHistory":
        t = training_frame(train).assign(
            LogSales=lambda x: np.log1p(x["Sales"]),
            LogCustomers=lambda x: np.log1p(x["Customers"]),
            LogBasket=lambda x: np.log(x["Sales"] / x["Customers"]),
        )
        cutoff = t["Date"].max() - pd.Timedelta(days=self.RECENT_DAYS)
        self.tables_ = [
            (["Store"], t.groupby("Store").agg(
                StoreMeanLogSales=("LogSales", "mean"),
                StoreStdLogSales=("LogSales", "std"),
                StoreMeanLogCustomers=("LogCustomers", "mean"),
                StoreMeanLogBasket=("LogBasket", "mean"),
            )),
            (["Store"], t[t["Date"] > cutoff].groupby("Store").agg(
                StoreRecentMeanLogSales=("LogSales", "mean"),
            )),
            (["Store", "DayOfWeek"], t.groupby(["Store", "DayOfWeek"]).agg(
                StoreDowMeanLogSales=("LogSales", "mean"),
            )),
            (["Store", "Promo"], t.groupby(["Store", "Promo"]).agg(
                StorePromoMeanLogSales=("LogSales", "mean"),
            )),
        ]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Look up each row's store summaries. Unseen groups stay NaN."""
        out = df
        for keys, table in self.tables_:
            out = out.join(table, on=keys)
        return out


# Final feature list for the model. Deliberately excluded:
#   Sales (target), Customers (unknown at forecast time), Date / Id (identifiers),
#   Store (identity is captured by StoreHistory; as a raw number it is a
#   meaningless ID that trees would split on arbitrarily),
#   Promo2SinceWeek/Year, PromoInterval, CompetitionOpenSince* (raw inputs
#   already turned into Promo2Active, Promo2Weeks, CompetitionMonthsOpen).
FEATURES = [
    # calendar
    "DayOfWeek", "Year", "Month", "DayOfMonth", "WeekOfYear", "DayOfYear",
    # known-in-advance flags
    "Promo", "StateHoliday", "SchoolHoliday",
    # store attributes
    "StoreType", "Assortment", "CompetitionDistance", "CompetitionDistanceMissing",
    "CompetitionMonthsOpen", "Promo2", "Promo2Active", "Promo2Weeks",
    # event proximity
    *[f"Days{d}{e}" for e in EVENTS for d in ("Since", "Until")],
    # store history (fitted on training period only)
    "StoreMeanLogSales", "StoreStdLogSales", "StoreMeanLogCustomers",
    "StoreMeanLogBasket", "StoreRecentMeanLogSales",
    "StoreDowMeanLogSales", "StorePromoMeanLogSales",
]
