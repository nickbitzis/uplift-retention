import numpy as np
import pytest

from uplift import evaluation as ev

# Hand-worked example. Sorted by score (6 down to 1):
#   treated  1 0 1 1 0 0
#   retained 1 0 1 0 1 0
# Qini(k) = R_t - R_c * N_t / N_c  ->  0, 1, 1, 2, 2, 0.5, 1
# Uplift(k) = (R_t/N_t - R_c/N_c) * k  ->  0, 0, 2, 3, 8/3, 5/6, 2
# Qini area (step 1/6) = (0.5 + 1 + 1.5 + 2 + 1.25 + 0.75) / 6 = 7/6, random area = 1/2, coefficient = 2/3.
SCORE = np.array([3, 6, 1, 4, 5, 2])
TREAT = np.array([1, 1, 0, 1, 0, 0])
RET = np.array([0, 1, 0, 1, 0, 1])


def test_qini_curve_hand_worked():
    x, q = ev.qini_curve(SCORE, TREAT, RET)
    assert np.allclose(q, [0, 1, 1, 2, 2, 0.5, 1])
    assert np.allclose(x, np.arange(7) / 6)


def test_uplift_curve_hand_worked():
    _, u = ev.uplift_curve(SCORE, TREAT, RET)
    assert np.allclose(u, [0, 0, 2, 3, 8 / 3, 5 / 6, 2])


def test_qini_area_and_coefficient_hand_worked():
    x, q = ev.qini_curve(SCORE, TREAT, RET)
    assert ev.area(x, q) == pytest.approx(7 / 6)
    assert ev.qini_coefficient(SCORE, TREAT, RET) == pytest.approx(2 / 3)


def test_uplift_auc_hand_worked():
    expected = (0 + 1 + 2.5 + (3 + 8 / 3) / 2 + (8 / 3 + 5 / 6) / 2 + (5 / 6 + 2) / 2) / 6
    assert ev.uplift_auc(SCORE, TREAT, RET) == pytest.approx(expected)


def test_area_trapezoid():
    assert ev.area(np.array([0, 1]), np.array([0, 2])) == pytest.approx(1.0)


def test_perfect_ranking_beats_random_and_reversed():
    rng = np.random.default_rng(0)
    n = 20_000
    eff = rng.random(n) < 0.3
    t = rng.integers(0, 2, n)
    r = np.where(t == 1, (rng.random(n) < np.where(eff, 0.9, 0.5)), rng.random(n) < 0.5).astype(int)
    good = ev.qini_coefficient(eff.astype(float) + rng.random(n) * 0.1, t, r)
    rand = ev.qini_coefficient(rng.random(n), t, r)
    bad = ev.qini_coefficient(-eff.astype(float) + rng.random(n) * 0.1, t, r)
    assert good > 300 and bad < -300 and abs(rand) < 120


def test_curve_endpoint_is_overall_incremental():
    _, q = ev.qini_curve(SCORE, TREAT, RET)
    # Overall: treated retained 2 of 3, control retained 1 of 3 -> 2 - 1 * 3 / 3 = 1
    assert q[-1] == pytest.approx(1.0)


def test_true_gain_curve():
    x, g = ev.true_gain_curve(np.array([0.1, 0.9, 0.5]), np.array([1, 1, -1]))
    assert g.tolist() == [0, 1, 0, 1]


def test_ties_keep_input_order():
    _, g = ev.true_gain_curve(np.array([1.0, 1.0, 1.0]), np.array([1, -1, 1]))
    assert g.tolist() == [0, 1, 0, 1]


def test_downsample_keeps_endpoints():
    x = np.linspace(0, 1, 1001)
    pts = ev.downsample(x, x * 10, 11)
    assert len(pts) == 11 and pts[0] == [0.0, 0.0] and pts[-1] == [1.0, 10.0]


def test_deciles():
    rng = np.random.default_rng(1)
    n = 1000
    s = rng.random(n)
    t = np.tile([0, 1], n // 2)
    r = np.where((t == 1) & (s > 0.9), 1, 0)
    d = ev.deciles(s, t, r, true_effect=(s > 0.9).astype(float))
    assert len(d) == 10 and sum(x["customers"] for x in d) == n
    assert d[0]["observed"] > 0.9 and d[0]["true"] > 0.9
    assert all(x["observed"] == 0 for x in d[2:])
