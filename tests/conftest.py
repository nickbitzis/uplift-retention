import numpy as np
import pandas as pd
import pytest

from uplift.pipeline import Study, StudyConfig

SMALL = StudyConfig(n=30_000, pilot_n=16_000, rollout_n=20_000, seed=7, max_iter=60)


@pytest.fixture(scope="session")
def study():
    return Study(SMALL).run()


def known_effect_data(n=6000, seed=0, effect=0.4, dog=False):
    """x0 > 0: the treatment cuts churn from 0.5 to 0.1 (effect 0.4). x0 <= 0: no effect.
    With dog=True, x1 > 0 customers instead churn more when treated (0.1 to 0.4)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    t = np.tile([0, 1], n // 2)
    rng.shuffle(t)
    p0 = np.full(n, 0.5)
    p1 = np.where(X[:, 0] > 0, 0.5 - effect, 0.5)
    if dog:
        p0 = np.where(X[:, 1] > 0, 0.1, 0.5)
        p1 = np.where(X[:, 1] > 0, 0.4, 0.5)
    y = (rng.random(n) < np.where(t == 1, p1, p0)).astype(int)
    return X, t, y


def frame(rows):
    base = {"customer_id": "C1", "tenure_months": 10, "plan": "standard", "price_increase": 0, "weekly_hours": 5.0,
            "usage_trend": 0.0, "support_tickets": 0, "payment_failures": 0, "primary_genre": "drama",
            "genre_count": 2, "device_count": 1, "promo_history": 0}
    return pd.DataFrame([{**base, **r} for r in rows])
