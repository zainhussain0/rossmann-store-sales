import numpy as np
import pytest

from src.metrics import rmspe


def test_hand_computed_value():
    # Both predictions are off by exactly 10%, so RMSPE = 0.1.
    assert rmspe([100, 200], [110, 180]) == pytest.approx(0.1)


def test_zero_sales_days_are_ignored():
    # The first day (true 0) would be undefined; only the 10% miss counts.
    assert rmspe([0, 100], [50, 90]) == pytest.approx(0.1)


def test_perfect_forecast_scores_zero():
    assert rmspe([5, 10, 15], [5, 10, 15]) == 0.0


def test_over_prediction_costs_more_than_under():
    # Same absolute miss of 50, but RMSPE is asymmetric in ratio terms:
    # 2x the truth costs 100%, while 1/2 the truth costs only 50%.
    assert rmspe([50], [100]) > rmspe([100], [50])


def test_all_zero_truth_raises():
    with pytest.raises(ValueError):
        rmspe([0, 0], [1, 2])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        rmspe(np.ones(3), np.ones(2))
