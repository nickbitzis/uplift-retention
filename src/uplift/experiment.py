"""Experiment design for the follow-up test: power analysis for the holdout, then a readout with intervals.

The test randomizes customers the policy selects: most get their recommended offer, a holdout share h gets
nothing. The metric is churn rate, holdout minus treated (positive = the policy keeps customers).

Sample size uses the normal approximation for two proportions with a pooled variance under the null:
    n = (z_{1-a/2} sqrt(pbar qbar (1/h + 1/(1-h))) + z_{1-b} sqrt(p_h q_h / h + p_t q_t / (1-h)))^2 / d^2
where n is the total test size, pbar = h p_h + (1-h) p_t, and d = p_h - p_t. With h = 0.5 this is the textbook
per-group formula times two (p = 0.50 vs 0.60, a = 0.05, power 0.80 gives 388 per group).
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

Z = NormalDist()


def sample_size(p_holdout: float, p_treated: float, holdout: float = 0.5, alpha: float = 0.05,
                power: float = 0.8) -> dict:
    """Customers needed in each arm to detect the difference between two churn rates."""
    d = p_holdout - p_treated
    if d == 0:
        raise ValueError("the two churn rates are equal: no effect to detect")
    h = holdout
    za, zb = Z.inv_cdf(1 - alpha / 2), Z.inv_cdf(power)
    pbar = h * p_holdout + (1 - h) * p_treated
    num = za * math.sqrt(pbar * (1 - pbar) * (1 / h + 1 / (1 - h))) + zb * math.sqrt(
        p_holdout * (1 - p_holdout) / h + p_treated * (1 - p_treated) / (1 - h)
    )
    total = (num / d) ** 2
    n_h, n_t = math.ceil(total * h), math.ceil(total * (1 - h))
    return {"holdout": n_h, "treated": n_t, "total": n_h + n_t, "holdout_share": h, "effect": round(d, 5)}


def achieved_power(p_holdout: float, p_treated: float, n_holdout: int, n_treated: int, alpha: float = 0.05) -> float:
    d = abs(p_holdout - p_treated)
    n = n_holdout + n_treated
    h = n_holdout / n
    pbar = h * p_holdout + (1 - h) * p_treated
    se0 = math.sqrt(pbar * (1 - pbar) * (1 / n_holdout + 1 / n_treated))
    se1 = math.sqrt(p_holdout * (1 - p_holdout) / n_holdout + p_treated * (1 - p_treated) / n_treated)
    za = Z.inv_cdf(1 - alpha / 2)
    return Z.cdf((d - za * se0) / se1)


def plan_test(p_holdout: float, p_treated: float, targeted_per_month: int,
              shares=(0.1, 0.2, 0.3, 0.5), alpha: float = 0.05, power: float = 0.8) -> list[dict]:
    """For each holdout share: customers needed and how many months of targeted volume that takes."""
    rows = []
    for h in shares:
        s = sample_size(p_holdout, p_treated, h, alpha, power)
        s["months"] = round(s["total"] / max(targeted_per_month, 1), 2)
        rows.append(s)
    return rows


def _ci(est: float, se: float, alpha: float = 0.05) -> list[float]:
    z = Z.inv_cdf(1 - alpha / 2)
    return [est - z * se, est + z * se]


def readout(y_holdout, y_treated, v_holdout=None, v_treated=None, alpha: float = 0.05) -> dict:
    """Churn rates, difference with a Wald interval and pooled z test, and optional net value per customer."""
    yh, yt = np.asarray(y_holdout, float), np.asarray(y_treated, float)
    nh, nt = len(yh), len(yt)
    ph, pt = yh.mean(), yt.mean()
    d = ph - pt
    se = math.sqrt(ph * (1 - ph) / nh + pt * (1 - pt) / nt)
    pool = (yh.sum() + yt.sum()) / (nh + nt)
    se0 = math.sqrt(pool * (1 - pool) * (1 / nh + 1 / nt))
    z = d / se0 if se0 > 0 else 0.0
    out = {
        "holdout": {"customers": nh, "churn_rate": round(ph, 5)},
        "treated": {"customers": nt, "churn_rate": round(pt, 5)},
        "churn_reduction": round(d, 5),
        "churn_reduction_ci": [round(x, 5) for x in _ci(d, se, alpha)],
        "z": round(z, 3),
        "p_value": float(2 * (1 - Z.cdf(abs(z)))),
        "incremental_retained": round(d * nt, 1),
        "incremental_retained_ci": [round(x * nt, 1) for x in _ci(d, se, alpha)],
    }
    if v_holdout is not None and v_treated is not None:
        vh, vt = np.asarray(v_holdout, float), np.asarray(v_treated, float)
        dv = vt.mean() - vh.mean()
        sev = math.sqrt(vt.var(ddof=1) / nt + vh.var(ddof=1) / nh)
        out["net_value_per_customer"] = round(dv, 3)
        out["net_value_per_customer_ci"] = [round(x, 3) for x in _ci(dv, sev, alpha)]
    return out
