"""Churn propensity and uplift learners, all on scikit-learn gradient boosting.

Sign convention everywhere: uplift = churn probability without the offer minus churn probability with it,
so a positive number means the offer keeps the customer.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .generator import GENRES
from .offers import PLANS

NUMERIC = (
    "tenure_months",
    "price_increase",
    "weekly_hours",
    "usage_trend",
    "support_tickets",
    "payment_failures",
    "genre_count",
    "device_count",
    "promo_history",
)
FEATURE_NAMES = (
    list(NUMERIC) + [f"plan_{p}" for p in PLANS] + [f"genre_{g}" for g in GENRES]
)


def matrix(df: pd.DataFrame) -> np.ndarray:
    """Feature matrix: numeric columns plus one-hot plan and primary genre. Ground truth columns are never read."""
    cols = [df[c].to_numpy(float) for c in NUMERIC]
    plan = df["plan"].to_numpy()
    genre = df["primary_genre"].to_numpy()
    cols += [(plan == p).astype(float) for p in PLANS]
    cols += [(genre == g).astype(float) for g in GENRES]
    return np.column_stack(cols)


def default_classifier(seed: int = 0, max_iter: int = 200, leaves: int = 31) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=leaves, min_samples_leaf=100, l2_regularization=1.0,
        early_stopping=False, random_state=seed,
    )


def default_regressor(seed: int = 0, max_iter: int = 150) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=False, random_state=seed,
    )


Factory = Callable[[], object]


class ChurnModel:
    """Plain churn propensity: P(churn | x) with no offer, trained on the control arm."""

    name = "churn_risk"

    def __init__(self, clf: Factory | None = None):
        self.clf = (clf or default_classifier)()

    def fit(self, X, t, y):
        self.clf.fit(X[t == 0], y[t == 0])
        return self

    def predict_churn(self, X) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]


class TLearner:
    """Two models: one on control, one on treated. Uplift is the difference of their predictions."""

    name = "t_learner"
    label = "T-learner (two models)"

    def __init__(self, clf: Factory | None = None):
        f = clf or default_classifier
        self.m0, self.m1 = f(), f()

    def fit(self, X, t, y):
        self.m0.fit(X[t == 0], y[t == 0])
        self.m1.fit(X[t == 1], y[t == 1])
        return self

    def predict_uplift(self, X) -> np.ndarray:
        return self.m0.predict_proba(X)[:, 1] - self.m1.predict_proba(X)[:, 1]


class SLearner:
    """One model with the treatment flag as a feature. Score each customer twice, with the flag off and on."""

    name = "s_learner"
    label = "S-learner (one model)"

    def __init__(self, clf: Factory | None = None):
        self.m = (clf or default_classifier)()

    def fit(self, X, t, y):
        self.m.fit(np.column_stack([X, t]), y)
        return self

    def predict_uplift(self, X) -> np.ndarray:
        n = len(X)
        p0 = self.m.predict_proba(np.column_stack([X, np.zeros(n)]))[:, 1]
        p1 = self.m.predict_proba(np.column_stack([X, np.ones(n)]))[:, 1]
        return p0 - p1


class XLearner:
    """X-learner (Kunzel et al. 2019).

    Stage 1: outcome models mu0 and mu1. Stage 2: impute each customer's effect using the other arm's model
    (treated: mu0(x) - y, control: y - mu1(x)) and regress it on x, once per arm. Stage 3: blend the two effect
    models with the propensity g, which is a known 0.5 here because the campaign was randomized.
    """

    name = "x_learner"
    label = "X-learner"

    def __init__(self, clf: Factory | None = None, reg: Factory | None = None, propensity: float = 0.5):
        f, r = clf or default_classifier, reg or default_regressor
        self.mu0, self.mu1 = f(), f()
        self.tau0, self.tau1 = r(), r()
        self.g = propensity

    def fit(self, X, t, y):
        X0, y0, X1, y1 = X[t == 0], y[t == 0], X[t == 1], y[t == 1]
        self.mu0.fit(X0, y0)
        self.mu1.fit(X1, y1)
        d1 = self.mu0.predict_proba(X1)[:, 1] - y1
        d0 = y0 - self.mu1.predict_proba(X0)[:, 1]
        self.tau1.fit(X1, d1)
        self.tau0.fit(X0, d0)
        return self

    def predict_uplift(self, X) -> np.ndarray:
        return self.g * self.tau0.predict(X) + (1 - self.g) * self.tau1.predict(X)


class ClassTransformation:
    """Class transformation (Jaskowski and Jaroszewicz 2012).

    With a 50/50 randomized treatment, define Z = 1 when (treated and retained) or (control and churned).
    Then P(Z=1 | x) = (1 + uplift(x)) / 2, so one ordinary classifier on Z gives uplift = 2 P(Z=1 | x) - 1.
    """

    name = "class_transform"
    label = "Class transformation"

    def __init__(self, clf: Factory | None = None):
        self.m = (clf or default_classifier)()

    @staticmethod
    def transform(t, y) -> np.ndarray:
        t, y = np.asarray(t), np.asarray(y)
        return ((t == 1) & (y == 0) | (t == 0) & (y == 1)).astype(int)

    def fit(self, X, t, y):
        share = float(np.mean(t))
        if abs(share - 0.5) > 0.05:
            raise ValueError(f"class transformation needs a 50/50 split, got {share:.2f} treated")
        self.m.fit(X, self.transform(t, y))
        return self

    def predict_uplift(self, X) -> np.ndarray:
        return 2 * self.m.predict_proba(X)[:, 1] - 1


LEARNERS = (TLearner, SLearner, XLearner, ClassTransformation)


class MultiOfferModel:
    """One outcome model per arm (control plus each offer), a T-learner with a shared control model.

    Trained on the main campaign (control and discount) plus the three-offer pilot.
    """

    def __init__(self, n_offers: int = 3, clf: Factory | None = None):
        f = clf or default_classifier
        self.models = [f() for _ in range(n_offers + 1)]

    def fit(self, X, arm, y):
        for a, m in enumerate(self.models):
            m.fit(X[arm == a], y[arm == a])
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Returns (p0, pk) where pk has one column per offer."""
        p = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models])
        return p[:, 0], p[:, 1:]


def clone_factory(est) -> Factory:
    return lambda: clone(est)
