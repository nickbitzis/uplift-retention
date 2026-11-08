import numpy as np
import pytest

from uplift.generator import GROUPS, GenConfig, generate, split
from uplift.learners import FEATURE_NAMES, NUMERIC, matrix


@pytest.fixture(scope="module")
def df():
    return generate(GenConfig(n=60_000, seed=11))


def test_every_customer_in_exactly_one_group(df):
    assert set(df["true_group"]) == set(GROUPS)
    assert df["true_group"].notna().all()


def test_groups_match_potential_outcomes(df):
    y0, yd = df["y0"], df["y_discount"]
    g = df["true_group"]
    assert ((g == "persuadable") == ((y0 == 1) & (yd == 0))).all()
    assert ((g == "sure_thing") == ((y0 == 0) & (yd == 0))).all()
    assert ((g == "lost_cause") == ((y0 == 1) & (yd == 1))).all()
    assert ((g == "sleeping_dog") == ((y0 == 0) & (yd == 1))).all()


def test_observed_churn_is_the_assigned_potential_outcome(df):
    expected = np.where(df["treated"] == 1, df["y_discount"], df["y0"])
    assert (df["churned"].to_numpy() == expected).all()


def test_multi_arm_design_uses_the_arm_outcome():
    m = generate(GenConfig(n=8000, seed=3, design="multi"))
    cols = np.column_stack([m["y0"], m["y_discount"], m["y_free_month"], m["y_downgrade"]])
    assert (m["churned"].to_numpy() == cols[np.arange(len(m)), m["arm"]]).all()
    shares = m["arm"].value_counts(normalize=True)
    assert all(abs(shares[a] - 0.25) < 0.02 for a in range(4))


def test_randomization_share(df):
    assert abs(df["treated"].mean() - 0.5) < 0.01


@pytest.mark.parametrize("col", NUMERIC)
def test_randomization_balances_features(df, col):
    t, c = df.loc[df["treated"] == 1, col], df.loc[df["treated"] == 0, col]
    smd = abs(t.mean() - c.mean()) / np.sqrt((t.var() + c.var()) / 2)
    assert smd < 0.03


def test_plan_mix_balanced(df):
    tab = df.groupby("treated")["plan"].value_counts(normalize=True).unstack()
    assert (tab.diff().abs().iloc[1] < 0.015).all()


def test_seeded_and_deterministic():
    a, b = generate(GenConfig(n=2000, seed=5)), generate(GenConfig(n=2000, seed=5))
    c = generate(GenConfig(n=2000, seed=6))
    assert a.equals(b)
    assert not a["churned"].equals(c["churned"])


def test_group_shares_are_plausible(df):
    s = df["true_group"].value_counts(normalize=True)
    assert 0.06 < s["persuadable"] < 0.12
    assert 0.03 < s["sleeping_dog"] < 0.08
    assert 0.08 < s["lost_cause"] < 0.14
    assert s["sure_thing"] > 0.65


def test_true_uplift_matches_realized_effect(df):
    realized = (df["y0"] - df["y_discount"]).mean()
    assert abs(df["true_uplift_discount"].mean() - realized) < 0.004


def test_discount_reduces_observed_churn(df):
    c = df.groupby("treated")["churned"].mean()
    assert c[0] - c[1] > 0.02


def test_sleeping_dogs_are_low_usage_long_tenure(df):
    dogs, rest = df[df["true_group"] == "sleeping_dog"], df[df["true_group"] != "sleeping_dog"]
    assert dogs["weekly_hours"].median() < rest["weekly_hours"].median()
    assert dogs["tenure_months"].median() > rest["tenure_months"].median()


def test_lost_causes_riskier_than_sure_things(df):
    g = df.groupby("true_group")["true_p0"].mean()
    assert g["lost_cause"] > 1.5 * g["sure_thing"]


def test_downgrade_does_nothing_on_basic(df):
    basic = df[df["plan"] == "basic"]
    assert (basic["y_downgrade"] == basic["y0"]).all()
    assert (basic["true_uplift_downgrade"] == 0).all()


def test_split_proportions_and_determinism(df):
    s1, s2 = split(df, 1), split(df, 1)
    assert s1.equals(s2)
    p = s1.value_counts(normalize=True)
    assert abs(p["train"] - 0.6) < 0.01 and abs(p["test"] - 0.2) < 0.01


def test_unknown_design_raises():
    with pytest.raises(ValueError):
        generate(GenConfig(n=10, design="bogus"))


def test_feature_matrix_ignores_ground_truth(df):
    small = df.head(500).copy()
    X1 = matrix(small)
    for col in ["y0", "y_discount", "true_group", "true_p0", "churned", "treated"]:
        small[col] = 0
    assert X1.shape == (500, len(FEATURE_NAMES))
    assert np.array_equal(X1, matrix(small))
