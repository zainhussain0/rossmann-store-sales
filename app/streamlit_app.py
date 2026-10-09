"""What-if explorer for the Rossmann 48-day sales forecast.

Run from the project root:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets us `import src`

import altair as alt
import pandas as pd
import streamlit as st

from src.data import load_all
from src.scenarios import compare, load_or_train

# Validated scores, from notebooks/02_modelling.ipynb (mean of 3 rolling folds).
MODEL_RMSPE, BASELINE_RMSPE = 0.106, 0.154

LEVERS = {
    "Run a promo (Mon-Fri)": ({"Promo": 1}, True),
    "Cancel planned promos": ({"Promo": 0}, False),
    "Close the store": ({"Open": 0}, False),
}

st.set_page_config(page_title="Rossmann what-if forecast", layout="wide")


@st.cache_data(show_spinner="Loading data...")
def data():
    return load_all()


@st.cache_resource(show_spinner="Loading model (first run trains it, about a minute)...")
def model():
    return load_or_train()


raw_train, raw_test = data()
mdl = model()
first_day, last_day = raw_test["Date"].min().date(), raw_test["Date"].max().date()

st.title("Rossmann store sales: what-if forecast")
st.caption(
    f"Daily sales forecast for {first_day:%d %b} to {last_day:%d %b %Y}, 48 days past the last known sales. "
    f"Validated RMSPE {MODEL_RMSPE:.3f}, against {BASELINE_RMSPE:.3f} for the best simple baseline."
)

with st.sidebar:
    st.header("Scenario")
    store = st.selectbox("Store", sorted(raw_test["Store"].unique()))
    lever = st.radio("Change", list(LEVERS))
    start, end = st.slider(
        "Days to change", min_value=first_day, max_value=last_day,
        value=(pd.Timestamp("2015-08-10").date(), pd.Timestamp("2015-08-14").date()),
        format="DD MMM",
    )
    changes, weekdays_only = LEVERS[lever]

try:
    daily, totals = compare(mdl, raw_train, raw_test, store, changes, start, end, weekdays_only)
except ValueError as e:
    st.error(str(e))
    st.stop()

edited = daily["Date"].between(pd.Timestamp(start), pd.Timestamp(end))
if (daily.loc[edited, ["Promo", "Open"]].values == daily.loc[edited, ["ScenarioPromo", "ScenarioOpen"]].values).all():
    st.warning("Nothing changed: the selected days already match this scenario. "
               "Try other dates (promos usually run every other week).")

c1, c2, c3 = st.columns(3)
c1.metric("Planned, 48 days", f"{totals['planned']:,.0f}")
c2.metric("Scenario, 48 days", f"{totals['scenario']:,.0f}")
c3.metric("Difference", f"{totals['difference']:+,.0f}", f"{totals['pct_change']:+.1%}")

# Last 6 weeks of actual sales for context, then both forecasts.
history = raw_train[(raw_train["Store"] == store) & (raw_train["Date"] > raw_train["Date"].max() - pd.Timedelta(weeks=6))]
lines = pd.concat([
    history[["Date", "Sales"]].assign(Series="Actual"),
    daily[["Date", "Planned"]].rename(columns={"Planned": "Sales"}).assign(Series="Planned"),
    daily[["Date", "Scenario"]].rename(columns={"Scenario": "Sales"}).assign(Series="Scenario"),
])
window = pd.DataFrame({"start": [pd.Timestamp(start)], "end": [pd.Timestamp(end) + pd.Timedelta(days=1)]})
chart = (
    alt.Chart(window).mark_rect(opacity=0.12).encode(x="start:T", x2="end:T")
    + alt.Chart(lines).mark_line(point=alt.OverlayMarkDef(size=18)).encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y("Sales:Q", title="Daily sales"),
        color=alt.Color("Series:N", scale=alt.Scale(
            domain=["Actual", "Planned", "Scenario"], range=["#8c8c8c", "#1f77b4", "#e4572e"])),
        strokeDash=alt.condition(alt.datum.Series == "Scenario", alt.value([5, 3]), alt.value([1, 0])),
        tooltip=["Series", alt.Tooltip("Date:T", format="%a %d %b"), alt.Tooltip("Sales:Q", format=",.0f")],
    )
).properties(height=380)
st.altair_chart(chart, use_container_width=True)
st.caption("Shaded: days you changed. Closed days are forecast as 0.")

with st.expander("Day-by-day"):
    shown = daily[daily["Difference"].abs() > 0.5].copy()
    shown["Date"] = shown["Date"].dt.strftime("%a %d %b")
    st.dataframe(
        shown[["Date", "Promo", "ScenarioPromo", "Open", "ScenarioOpen", "Planned", "Scenario", "Difference"]]
        .style.format({"Planned": "{:,.0f}", "Scenario": "{:,.0f}", "Difference": "{:+,.0f}"}),
        hide_index=True, width="stretch",
    )
    st.caption("Only days whose forecast changed. Neighbouring days can move too: "
               "the model uses days until / since the nearest promo and closure.")

st.info(
    "**How to read this.** The scenario shows what the model *expects*, learned from 2.5 years of "
    "history. It is not a causal estimate: promos weren't assigned at random, so a predicted uplift "
    "mixes the promo's real effect with whatever usually came with promo days. "
    "Weekend promos are blocked because none ran in the training data."
)
