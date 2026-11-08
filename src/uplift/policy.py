"""Policy value math. A policy assigns each customer one offer index (0, 1, 2 in OFFER_KEYS order) or -1 for none.

For a customer who gets offer k, compared with getting nothing:
    incremental retained = y0 - y_k                          (1 saved, 0 no change, -1 sleeping dog woken)
    net value            = (1 - y_k) * CLV_k - cost_k - (1 - y0) * CLV_0
Customers with no offer contribute 0. With model estimates in place of the true outcomes this becomes the
expected net value used for targeting: (1 - p_k) * CLV_k - cost_k - (1 - p0) * CLV_0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .generator import GROUPS
from .offers import OFFER_KEYS, offer_economics

NONE = -1


def potential_outcomes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y0 = df["y0"].to_numpy(int)
    yk = np.column_stack([df[f"y_{k}"].to_numpy(int) for k in OFFER_KEYS])
    return y0, yk


def expected_values(p0: np.ndarray, pk: np.ndarray, econ: dict) -> np.ndarray:
    """Expected net value of each offer for each customer, shape (n, 3). Ineligible offers are -inf."""
    v = (1 - pk) * econ["clv_k"] - econ["cost"] - (1 - p0)[:, None] * econ["clv0"][:, None]
    return np.where(econ["eligible"], v, -np.inf)


def top_k(score: np.ndarray, k: int, offer: int = 0) -> np.ndarray:
    """Give `offer` to the k highest scores."""
    assign = np.full(len(score), NONE)
    if k > 0:
        assign[np.argsort(-np.asarray(score, float), kind="stable")[:k]] = offer
    return assign


def evaluate(df: pd.DataFrame, assign: np.ndarray, econ: dict | None = None) -> dict:
    """True outcome of a policy on synthetic customers, using every customer's potential outcomes."""
    econ = econ or offer_economics(df["plan"].to_numpy(), df["tenure_months"].to_numpy())
    y0, yk = potential_outcomes(df)
    assign = np.asarray(assign)
    on = assign != NONE
    idx = np.where(on)[0]
    k = assign[idx]
    y_off = yk[idx, k]
    y_base = y0[idx]
    cost = econ["cost"][idx, k]
    value = (1 - y_off) * econ["clv_k"][idx, k] - cost - (1 - y_base) * econ["clv0"][idx]
    group = np.select(
        [(y_base == 1) & (y_off == 0), (y_base == 0) & (y_off == 0), (y_base == 1) & (y_off == 1)],
        list(GROUPS[:3]),
        default=GROUPS[3],
    )
    groups = {g: int((group == g).sum()) for g in GROUPS}
    offers = {key: int((k == i).sum()) for i, key in enumerate(OFFER_KEYS)}
    return {
        "targeted": int(on.sum()),
        "cost": round(float(cost.sum()), 2),
        "incremental_retained": int((y_base - y_off).sum()),
        "net_value": round(float(value.sum()), 2),
        "groups": groups,
        "offers": offers,
        # Money spent on customers whose outcome the offer did not change, and churn the offer caused.
        "wasted_cost": round(float(cost[group != "persuadable"].sum()), 2),
    }


def expected(assign: np.ndarray, values: np.ndarray, econ: dict) -> dict:
    """What the models expect a policy to return (no ground truth needed)."""
    idx = np.where(np.asarray(assign) != NONE)[0]
    k = np.asarray(assign)[idx]
    return {
        "expected_net_value": round(float(values[idx, k].sum()), 2),
        "cost": round(float(econ["cost"][idx, k].sum()), 2),
    }
