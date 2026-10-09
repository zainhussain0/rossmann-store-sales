import pandas as pd

from src.data import (load_store, load_test, merge_store, parse_promo_months,
                      training_frame)


def test_september_is_parsed():
    # Regression test for the 'Sept' vs 'Sep' trap.
    assert parse_promo_months("Mar,Jun,Sept,Dec") == (3, 6, 9, 12)


def test_no_promo_gives_empty_tuple():
    assert parse_promo_months("None") == ()
    assert parse_promo_months(float("nan")) == ()


def test_store_has_no_unexpected_nulls():
    s = load_store()
    for col in ["CompetitionDistance", "Promo2SinceWeek", "PromoInterval"]:
        assert s[col].notna().all()
    assert len(s) == s["Store"].nunique() == 1115


def test_test_open_has_no_nulls():
    assert load_test()["Open"].notna().all()


def test_merge_keeps_row_count():
    t = load_test()
    assert len(merge_store(t, load_store())) == len(t)


def test_training_frame_drops_closed_days():
    df = pd.DataFrame({"Open": [1, 0, 1], "Sales": [100, 0, 0]})
    assert len(training_frame(df)) == 1