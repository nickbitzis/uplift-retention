import numpy as np
import pytest

from tests.conftest import known_effect_data
from uplift.learners import (
    LEARNERS,
    ChurnModel,
    ClassTransformation,
    MultiOfferModel,
    default_classifier,
    default_regressor,
)


def small_clf():
    return default_classifier(0, max_iter=60)


def make(cls):
    if cls.name == "x_learner":
        return cls(small_clf, lambda: default_regressor(0, max_iter=60))
    return cls(small_clf)


@pytest.fixture(scope="module")
def fitted():
    X, t, y = known_effect_data()
    Xt = np.random.default_rng(9).normal(size=(4000, 4))
    return {cls.name: make(cls).fit(X, t, y).predict_uplift(Xt) for cls in LEARNERS}, Xt


@pytest.mark.parametrize("name", [c.name for c in LEARNERS])
def test_learner_finds_effect_where_it_exists(fitted, name):
    u, Xt = fitted
    assert 0.25 < u[name][Xt[:, 0] > 0].mean() < 0.55


@pytest.mark.parametrize("name", [c.name for c in LEARNERS])
def test_learner_finds_no_effect_where_there_is_none(fitted, name):
    u, Xt = fitted
    assert abs(u[name][Xt[:, 0] <= 0].mean()) < 0.1


@pytest.mark.parametrize("name", [c.name for c in LEARNERS])
def test_learner_ranks_treatable_customers_first(fitted, name):
    u, Xt = fitted
    top = np.argsort(-u[name])[: len(u[name]) // 4]
    assert (Xt[top, 0] > 0).mean() > 0.85


@pytest.mark.parametrize("cls", LEARNERS)
def test_learner_detects_sleeping_dogs(cls):
    X, t, y = known_effect_data(seed=4, dog=True)
    u = make(cls).fit(X, t, y).predict_uplift(X)
    assert u[X[:, 1] > 0].mean() < -0.15
    assert abs(u[X[:, 1] <= 0].mean()) < 0.1


def test_class_transformation_truth_table():
    t = np.array([1, 1, 0, 0])
    y = np.array([0, 1, 0, 1])
    assert ClassTransformation.transform(t, y).tolist() == [1, 0, 0, 1]


def test_class_transformation_needs_balanced_treatment():
    X = np.zeros((100, 2))
    t = np.r_[np.ones(80), np.zeros(20)].astype(int)
    with pytest.raises(ValueError):
        ClassTransformation(small_clf).fit(X, t, np.zeros(100, int))


def test_churn_model_uses_control_arm_only():
    X, t, y = known_effect_data(seed=2)
    y = np.where(t == 1, 1, y)  # treated all churn; a control-only model must not see that
    p = ChurnModel(small_clf).fit(X, t, y).predict_churn(X)
    assert abs(p.mean() - 0.5) < 0.05


def test_multi_offer_model_recovers_per_arm_effects():
    rng = np.random.default_rng(1)
    n = 12_000
    X = rng.normal(size=(n, 3))
    arm = rng.integers(0, 4, n)
    p = np.choose(arm, [np.full(n, 0.4), np.full(n, 0.3), np.where(X[:, 0] > 0, 0.1, 0.4), np.full(n, 0.45)])
    y = (rng.random(n) < p).astype(int)
    p0, pk = MultiOfferModel(3, small_clf).fit(X, arm, y).predict(X)
    up = p0[:, None] - pk
    assert abs(up[:, 0].mean() - 0.1) < 0.04
    assert up[X[:, 0] > 0, 1].mean() > 0.2
    assert up[:, 2].mean() < 0
