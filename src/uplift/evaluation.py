"""Qini and uplift curves, their areas, and uplift by decile.

All curves use retention (1 - churned) as the outcome, so the curve rises when the model ranks customers the
offer keeps ahead of customers it does not. Customers are sorted by score, highest first; ties keep input order.

For the top k customers with N_t treated, N_c control, and R_t, R_c retained in each arm:
    Qini(k)   = R_t - R_c * N_t / N_c       incremental retained, in units of the treated group
    Uplift(k) = (R_t / N_t - R_c / N_c) * k  incremental retained if all k got the offer
Both are 0 at k = 0, and a term is 0 while its arm has no customers yet.
"""

from __future__ import annotations

import numpy as np


def _cumulative(score, treated, retained):
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    t = np.asarray(treated)[order].astype(float)
    r = np.asarray(retained)[order].astype(float)
    nt = np.concatenate([[0.0], np.cumsum(t)])
    nc = np.concatenate([[0.0], np.cumsum(1 - t)])
    rt = np.concatenate([[0.0], np.cumsum(r * t)])
    rc = np.concatenate([[0.0], np.cumsum(r * (1 - t))])
    return nt, nc, rt, rc


def qini_curve(score, treated, retained) -> tuple[np.ndarray, np.ndarray]:
    """Returns (fraction targeted, Qini value) at every k from 0 to n."""
    nt, nc, rt, rc = _cumulative(score, treated, retained)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = rt - np.where(nc > 0, rc * nt / np.where(nc > 0, nc, 1), 0.0)
    x = np.arange(len(q)) / (len(q) - 1)
    return x, q


def uplift_curve(score, treated, retained) -> tuple[np.ndarray, np.ndarray]:
    nt, nc, rt, rc = _cumulative(score, treated, retained)
    k = np.arange(len(nt), dtype=float)
    rate_t = np.where(nt > 0, rt / np.where(nt > 0, nt, 1), 0.0)
    rate_c = np.where(nc > 0, rc / np.where(nc > 0, nc, 1), 0.0)
    u = np.where((nt > 0) & (nc > 0), (rate_t - rate_c) * k, 0.0)
    x = k / (len(k) - 1)
    return x, u


def area(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoid rule."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def qini_coefficient(score, treated, retained) -> float:
    """Area between the Qini curve and the straight line from (0, 0) to (1, Qini(n)): random targeting."""
    x, q = qini_curve(score, treated, retained)
    return area(x, q) - q[-1] / 2


def uplift_auc(score, treated, retained) -> float:
    x, u = uplift_curve(score, treated, retained)
    return area(x, u)


def true_gain_curve(score, true_effect) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic data only: true incremental retained if the top k got the offer (effect = y0 - y_offer)."""
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    g = np.concatenate([[0.0], np.cumsum(np.asarray(true_effect, float)[order])])
    return np.arange(len(g)) / (len(g) - 1), g


def downsample(x: np.ndarray, y: np.ndarray, points: int = 101) -> list[list[float]]:
    idx = np.unique(np.round(np.linspace(0, len(x) - 1, points)).astype(int))
    return [[round(float(x[i]), 4), round(float(y[i]), 2)] for i in idx]


def deciles(score, treated, retained, true_effect=None, bins: int = 10) -> list[dict]:
    """Uplift by score decile (decile 1 = highest scores). Observed uplift is retention treated minus control."""
    score = np.asarray(score, float)
    order = np.argsort(-score, kind="stable")
    chunks = np.array_split(order, bins)
    t, r = np.asarray(treated), np.asarray(retained)
    out = []
    for i, idx in enumerate(chunks, start=1):
        tt, rr = t[idx], r[idx]
        obs = (rr[tt == 1].mean() if (tt == 1).any() else 0.0) - (rr[tt == 0].mean() if (tt == 0).any() else 0.0)
        row = {
            "decile": i,
            "customers": int(len(idx)),
            "predicted": round(float(score[idx].mean()), 4),
            "observed": round(float(obs), 4),
        }
        if true_effect is not None:
            row["true"] = round(float(np.asarray(true_effect, float)[idx].mean()), 4)
        out.append(row)
    return out
