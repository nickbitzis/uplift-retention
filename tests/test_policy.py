import numpy as np
import pytest

from tests.conftest import frame
from uplift import policy as pol
from uplift.offers import CONTACT_COST, offer_economics

# Standard plan, 10 months: price 15.49, remaining months 12 + 4 = 16, CLV = 0.5 * 15.49 * 16 = 123.92.
CLV = 0.5 * 15.49 * 16
DISCOUNT = 0.75 * 15.49 + CONTACT_COST


def four_groups():
    df = frame([{}, {}, {}, {}])
    df["y0"] = [1, 0, 1, 0]
    df["y_discount"] = [0, 0, 1, 1]  # persuadable, sure thing, lost cause, sleeping dog
    df["y_free_month"] = df["y_discount"]
    df["y_downgrade"] = df["y0"]
    return df


def test_offer_economics_standard_plan():
    e = offer_economics(np.array(["standard"]), np.array([10]))
    assert e["clv0"][0] == pytest.approx(CLV)
    assert e["cost"][0].tolist() == pytest.approx([DISCOUNT, 15.49 + CONTACT_COST, CONTACT_COST])
    assert e["clv_k"][0, 2] == pytest.approx(0.5 * 8.99 * 16)


def test_clv_caps_remaining_months():
    e = offer_economics(np.array(["premium"]), np.array([200]))
    assert e["clv0"][0] == pytest.approx(0.5 * 22.99 * 36)


def test_downgrade_not_eligible_on_basic():
    e = offer_economics(np.array(["basic", "standard"]), np.array([5, 5]))
    assert e["eligible"][:, 2].tolist() == [False, True]


def test_policy_value_hand_worked():
    df = four_groups()
    r = pol.evaluate(df, np.zeros(4, int))
    # persuadable +CLV - c, sure thing -c, lost cause -c, sleeping dog -c - CLV  ->  total -4c
    assert r["net_value"] == pytest.approx(-4 * DISCOUNT, abs=0.01)
    assert r["incremental_retained"] == 0
    assert r["groups"] == {"persuadable": 1, "sure_thing": 1, "lost_cause": 1, "sleeping_dog": 1}
    assert r["wasted_cost"] == pytest.approx(3 * DISCOUNT, abs=0.01)


def test_only_persuadable_policy_value():
    r = pol.evaluate(four_groups(), np.array([0, -1, -1, -1]))
    assert r["net_value"] == pytest.approx(CLV - DISCOUNT, abs=0.01)
    assert r["targeted"] == 1 and r["incremental_retained"] == 1


def test_sleeping_dog_costs_its_clv():
    r = pol.evaluate(four_groups(), np.array([-1, -1, -1, 0]))
    assert r["net_value"] == pytest.approx(-CLV - DISCOUNT, abs=0.01)
    assert r["incremental_retained"] == -1


def test_no_offer_is_zero():
    r = pol.evaluate(four_groups(), np.full(4, -1))
    assert r["net_value"] == 0 and r["cost"] == 0 and r["targeted"] == 0


def test_groups_use_the_assigned_offer():
    # Under the downgrade (y_downgrade == y0) everyone is a sure thing or a lost cause.
    r = pol.evaluate(four_groups(), np.full(4, 2))
    assert r["groups"]["persuadable"] == 0 and r["groups"]["sleeping_dog"] == 0
    assert r["offers"]["downgrade"] == 4


def test_expected_values_formula():
    e = offer_economics(np.array(["standard"]), np.array([10]))
    v = pol.expected_values(np.array([0.5]), np.array([[0.3, 0.5, 0.5]]), e)
    assert v[0, 0] == pytest.approx(0.7 * CLV - DISCOUNT - 0.5 * CLV)
    assert v[0, 1] == pytest.approx(-(15.49 + CONTACT_COST))


def test_expected_values_ineligible_is_minus_inf():
    e = offer_economics(np.array(["basic"]), np.array([10]))
    v = pol.expected_values(np.array([0.5]), np.array([[0.4, 0.4, 0.0]]), e)
    assert v[0, 2] == -np.inf


def test_top_k():
    a = pol.top_k(np.array([0.1, 0.9, 0.5, 0.7]), 2, offer=1)
    assert a.tolist() == [-1, 1, -1, 1]
    assert (pol.top_k(np.array([1.0, 2.0]), 0) == -1).all()
