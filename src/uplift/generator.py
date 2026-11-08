"""Synthetic subscribers of Northlight, a made-up streaming service, with a known causal structure.

Each subscriber gets features, then three hidden draws decide their potential outcomes:

    churn with no offer:     Y0 = 1 with probability p0(x)
    if Y0 = 1, offer k saves them   with probability save_k(x)     (shared uniform U_save across offers)
    if Y0 = 0, offer k triggers churn with probability trigger_k(x) (shared uniform U_trig across offers)

So for the discount offer every subscriber falls in exactly one of four groups:
    persuadable   Y0 = 1, Y_discount = 0   the offer prevents churn
    sure thing    Y0 = 0, Y_discount = 0   stays anyway
    lost cause    Y0 = 1, Y_discount = 1   churns anyway
    sleeping dog  Y0 = 0, Y_discount = 1   the offer reminds them to cancel

The historical campaign sent the discount to a random 50% (`treated`), so effects are identifiable from the
observed `churned` column alone. A separate pilot randomizes all three offers for the multi-offer models.
Columns starting with `true_` or `y_` are ground truth and never used for training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .offers import PLANS

GENRES = ("drama", "comedy", "sports", "kids", "documentary", "anime", "reality")
GROUPS = ("persuadable", "sure_thing", "lost_cause", "sleeping_dog")
FEATURES = (
    "tenure_months",
    "plan",
    "price_increase",
    "weekly_hours",
    "usage_trend",
    "support_tickets",
    "payment_failures",
    "primary_genre",
    "genre_count",
    "device_count",
    "promo_history",
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class GenConfig:
    n: int = 200_000
    seed: int = 42
    # "binary": control vs discount, 50/50. "multi": control, discount, free month, downgrade, 25% each.
    design: str = "binary"
    treat_share: float = 0.5


def features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    plan = rng.choice(np.array(PLANS), size=n, p=[0.35, 0.45, 0.20])
    tenure = np.clip(np.round(rng.gamma(1.4, 14.0, n)), 1, 96).astype(int)
    price_increase = (rng.random(n) < np.where(plan == "basic", 0.10, 0.30)).astype(int)
    hours = np.round(np.clip(rng.lognormal(np.log(6.0), 0.75, n), 0.2, 50.0), 1)
    trend = np.round(np.clip(rng.normal(-0.05, 0.28, n), -1.0, 1.0), 3)
    tickets = rng.poisson(0.35 + 0.8 * (trend < -0.3) + 0.3 * price_increase)
    failures = rng.poisson(np.where(plan == "basic", 0.25, 0.12))
    genre = rng.choice(np.array(GENRES), size=n, p=[0.24, 0.18, 0.14, 0.14, 0.10, 0.10, 0.10])
    genre_count = 1 + rng.binomial(5, 0.35, n)
    devices = np.clip(1 + rng.poisson(np.where(plan == "premium", 1.6, 0.8) + 0.6 * (genre == "kids")), 1, 6)
    promos = np.minimum(rng.poisson(0.6, n), 5)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n)],
            "tenure_months": tenure,
            "plan": plan,
            "price_increase": price_increase,
            "weekly_hours": hours,
            "usage_trend": trend,
            "support_tickets": tickets,
            "payment_failures": failures,
            "primary_genre": genre,
            "genre_count": genre_count,
            "device_count": devices,
            "promo_history": promos,
        }
    )


def true_probabilities(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """The hidden structural model: p0 plus save and trigger probabilities for each offer."""
    t = df["tenure_months"].to_numpy(float)
    hours = df["weekly_hours"].to_numpy(float)
    trend = df["usage_trend"].to_numpy(float)
    tickets = df["support_tickets"].to_numpy(float)
    fail = df["payment_failures"].to_numpy(float)
    pinc = df["price_increase"].to_numpy(float)
    dev = df["device_count"].to_numpy(float)
    promo = df["promo_history"].to_numpy(float)
    gcount = df["genre_count"].to_numpy(float)
    plan = df["plan"].to_numpy()
    kids = (df["primary_genre"].to_numpy() == "kids").astype(float)
    sports = (df["primary_genre"].to_numpy() == "sports").astype(float)
    premium = (plan == "premium").astype(float)
    basic = plan == "basic"
    decline = np.maximum(-trend, 0.0)
    collapse = np.maximum(-trend - 0.45, 0.0)
    low_use = (hours < 2.5).astype(float)
    long_t = np.minimum(t, 60.0) / 60.0

    p0 = _sigmoid(
        -2.05
        + 0.85 * fail
        + 2.2 * decline
        + 0.30 * tickets
        + 0.95 * pinc
        - 0.65 * (np.log(hours) - np.log(6.0))
        - 0.012 * np.minimum(t, 60.0)
        - 0.18 * (dev - 1.0)
        - 0.45 * kids
        + 0.30 * sports
    )
    # Discount: works on price-sensitive customers and deal seekers, not on people who stopped watching or
    # whose card keeps failing.
    save_d = _sigmoid(-0.9 + 1.9 * pinc + 0.6 * premium + 0.35 * promo - 1.3 * (fail > 0) - 3.5 * collapse)
    # Sleeping dogs: long-tenure, low-usage, stable subscribers who forgot they pay. A promo email reminds them.
    sleepy = -3.4 + 2.2 * low_use + 1.6 * long_t - 0.7 * (dev > 2) - 0.9 * kids - 0.6 * decline
    trig_d = _sigmoid(sleepy)
    # Free month: brings lapsing viewers back to the catalog, so it helps declining usage more than price.
    save_f = _sigmoid(-1.2 + 3.0 * decline + 0.25 * (gcount >= 3) - 1.2 * (fail > 0) - 2.0 * collapse)
    trig_f = _sigmoid(sleepy - 0.3)
    # Downgrade: fixes affordability (payment failures, price increases). Not available on Basic.
    save_g = np.where(basic, 0.0, _sigmoid(-1.2 + 2.6 * (fail > 0) + 0.6 * pinc - 1.0 * decline))
    trig_g = np.where(basic, 0.0, _sigmoid(sleepy - 1.4))
    return {
        "p0": p0,
        "save": np.column_stack([save_d, save_f, save_g]),
        "trigger": np.column_stack([trig_d, trig_f, trig_g]),
    }


def generate(cfg: GenConfig | None = None) -> pd.DataFrame:
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(cfg.seed)
    df = features(cfg.n, rng)
    pr = true_probabilities(df)
    u0, us, ut = rng.random(cfg.n), rng.random(cfg.n), rng.random(cfg.n)
    y0 = u0 < pr["p0"]
    yk = np.where(y0[:, None], us[:, None] >= pr["save"], ut[:, None] < pr["trigger"])

    if cfg.design == "binary":
        arm = np.where(rng.random(cfg.n) < cfg.treat_share, 1, 0)
    elif cfg.design == "multi":
        arm = rng.integers(0, 4, cfg.n)
    else:
        raise ValueError(f"unknown design {cfg.design!r}")
    ypot = np.column_stack([y0, yk])
    churned = ypot[np.arange(cfg.n), arm]

    yd = yk[:, 0]
    group = np.select([y0 & ~yd, ~y0 & ~yd, y0 & yd, ~y0 & yd], list(GROUPS), default="")
    df["arm"] = arm
    df["treated"] = (arm > 0).astype(int)
    df["churned"] = churned.astype(int)
    df["y0"] = y0.astype(int)
    df["y_discount"] = yk[:, 0].astype(int)
    df["y_free_month"] = yk[:, 1].astype(int)
    df["y_downgrade"] = yk[:, 2].astype(int)
    df["true_group"] = group
    df["true_p0"] = pr["p0"]
    # Expected churn reduction from each offer: P(save) x P(would churn) - P(trigger) x P(would stay).
    for i, k in enumerate(("discount", "free_month", "downgrade")):
        df[f"true_uplift_{k}"] = pr["save"][:, i] * pr["p0"] - pr["trigger"][:, i] * (1 - pr["p0"])
    return df


def split(df: pd.DataFrame, seed: int = 42, valid: float = 0.2, test: float = 0.2) -> pd.Series:
    """Random train / valid / test labels (the campaign was randomized, so a random split is the right one)."""
    u = np.random.default_rng(seed + 1).random(len(df))
    return pd.Series(np.where(u < test, "test", np.where(u < test + valid, "valid", "train")), index=df.index)
