import math

import numpy as np
import pytest

from uplift import experiment as exp


def test_sample_size_textbook_value():
    # Two proportions 0.50 vs 0.60, alpha 0.05 two-sided, power 0.80: 387.3 -> 388 per group.
    s = exp.sample_size(0.6, 0.5, holdout=0.5)
    assert s["holdout"] == 388 and s["treated"] == 388


def test_sample_size_symmetric_in_direction():
    a, b = exp.sample_size(0.3, 0.2), exp.sample_size(0.2, 0.3)
    assert a["total"] == b["total"]


def test_smaller_holdout_needs_more_customers():
    totals = [exp.sample_size(0.3, 0.2, h)["total"] for h in (0.5, 0.3, 0.1)]
    assert totals[0] < totals[1] < totals[2]


def test_bigger_effect_needs_fewer():
    assert exp.sample_size(0.4, 0.2)["total"] < exp.sample_size(0.3, 0.2)["total"]


def test_achieved_power_at_planned_size():
    s = exp.sample_size(0.25, 0.2, holdout=0.2)
    p = exp.achieved_power(0.25, 0.2, s["holdout"], s["treated"])
    assert 0.8 <= p < 0.81


def test_zero_effect_raises():
    with pytest.raises(ValueError):
        exp.sample_size(0.2, 0.2)


def test_plan_test_months():
    rows = exp.plan_test(0.3, 0.2, targeted_per_month=100, shares=(0.5,))
    assert rows[0]["months"] == pytest.approx(rows[0]["total"] / 100, abs=0.01)


def test_readout_hand_worked():
    yh = np.r_[np.ones(30), np.zeros(70)]
    yt = np.r_[np.ones(20), np.zeros(80)]
    r = exp.readout(yh, yt)
    se = math.sqrt(0.21 / 100 + 0.16 / 100)
    assert r["churn_reduction"] == pytest.approx(0.1)
    assert r["churn_reduction_ci"][0] == pytest.approx(0.1 - 1.959964 * se, abs=1e-4)
    assert r["z"] == pytest.approx(0.1 / math.sqrt(0.25 * 0.75 * 0.02), abs=1e-3)
    assert r["p_value"] == pytest.approx(0.1025, abs=1e-3)
    assert r["incremental_retained"] == pytest.approx(10.0)


def test_readout_interval_covers_truth_most_of_the_time():
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(200):
        yh, yt = rng.random(800) < 0.3, rng.random(3200) < 0.22
        lo, hi = exp.readout(yh, yt)["churn_reduction_ci"]
        hits += lo <= 0.08 <= hi
    assert 0.91 <= hits / 200 <= 0.99


def test_readout_net_value_interval():
    rng = np.random.default_rng(1)
    r = exp.readout(np.zeros(50), np.zeros(50), rng.normal(10, 2, 500), rng.normal(15, 2, 500))
    lo, hi = r["net_value_per_customer_ci"]
    assert lo < r["net_value_per_customer"] < hi and lo > 4
