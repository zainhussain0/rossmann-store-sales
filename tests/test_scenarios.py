import numpy as np
import pandas as pd
import pytest

from src.data import load_all
from src.features import prepare
from src.models import GradientBoostingModel
from src.scenarios import compare, store_forecast


@pytest.fixture(scope="module")
def setup():
    # Small and fast: 20 stores, 20 trees. Checks wiring, not accuracy.
    raw_train, raw_test = load_all()
    raw_train = raw_train[raw_train["Store"] <= 20]
    raw_test = raw_test[raw_test["Store"] <= 20]
    train, test = prepare(raw_train, raw_test)
    model = GradientBoostingModel({"max_iter": 20}).fit(train)
    return model, raw_train, raw_test, test


def test_unchanged_scenario_matches_full_pipeline(setup):
    # Rebuilding one store's features on its own must give the same forecast
    # as building them for all stores at once (catches category-code drift).
    model, raw_train, raw_test, test = setup
    fc = store_forecast(model, raw_train, raw_test, store=1)
    expected = model.predict(test[test["Store"] == 1])
    np.testing.assert_allclose(fc["Forecast"].values, expected)


def test_closing_days_forecasts_zero(setup):
    model, raw_train, raw_test, _ = setup
    daily, totals = compare(model, raw_train, raw_test, 1, {"Open": 0}, "2015-08-03", "2015-08-05")
    edited = daily["Date"].between("2015-08-03", "2015-08-05")
    assert (daily.loc[edited, "Scenario"] == 0).all()
    assert totals["difference"] < 0


def test_rejects_non_levers_weekend_promos_and_unknown_stores(setup):
    model, raw_train, raw_test, _ = setup
    with pytest.raises(ValueError, match="lever"):
        store_forecast(model, raw_train, raw_test, 1, {"SchoolHoliday": 1}, "2015-08-03", "2015-08-05")
    with pytest.raises(ValueError, match="weekends"):
        store_forecast(model, raw_train, raw_test, 1, {"Promo": 1}, "2015-08-08", "2015-08-08")  # Saturday
    with pytest.raises(ValueError, match="not in the forecast window"):
        store_forecast(model, raw_train, raw_test, 2)  # store 2 isn't in test.csv


def test_weekdays_only_skips_the_weekend(setup):
    model, raw_train, raw_test, _ = setup
    # 2015-08-10 (Mon) to 2015-08-16 (Sun): a full week would be rejected without the flag.
    fc = store_forecast(model, raw_train, raw_test, 1, {"Promo": 1}, "2015-08-10", "2015-08-16",
                        weekdays_only=True)
    week = fc[fc["Date"].between("2015-08-10", "2015-08-16")]
    assert week.loc[week["DayOfWeek"] <= 5, "Promo"].eq(1).all()
    assert week.loc[week["DayOfWeek"] >= 6, "Promo"].eq(0).all()
