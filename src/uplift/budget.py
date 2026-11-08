"""Budget allocation across offers: at most one offer per customer, total cost within budget, maximize value.

This is a multiple-choice knapsack. The production method is greedy by expected incremental value per dollar,
which is fast enough to re-run on every slider move. A linear program relaxation (scipy HiGHS) gives an upper
bound on what any allocation could achieve, so the gap between the two says how much greedy leaves behind.
"""

from __future__ import annotations

import itertools

import numpy as np

try:  # scipy ships with scikit-learn, but keep the module usable without it.
    from scipy.optimize import LinearConstraint, linprog, milp
    from scipy.sparse import csr_matrix, vstack

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False

NONE = -1


def greedy_naive(values: np.ndarray, costs: np.ndarray, budget: float) -> np.ndarray:
    """Rank every (customer, offer) pair by value per dollar and take the first offer that fits per customer.

    Kept for comparison only: a near-free offer with a small value has a huge ratio, so it locks the customer
    out of a better offer. On the default run this loses about a third of the value."""
    n, k = values.shape
    v, c = values.ravel(), costs.ravel()
    cand = np.where(np.isfinite(v) & (v > 0))[0]
    ratio = v[cand] / np.maximum(c[cand], 1e-9)
    order = cand[np.lexsort((-v[cand], -ratio))]
    assign = np.full(n, NONE)
    spent = 0.0
    for j in order:
        i, offer = divmod(int(j), k)
        if assign[i] != NONE or spent + c[j] > budget + 1e-9:
            continue
        assign[i] = offer
        spent += c[j]
    return assign


def _hull(vals: np.ndarray, costs: np.ndarray) -> list[tuple[int, float, float]]:
    """Upper convex hull of one customer's options, starting from "no offer" at (0, 0).

    Returns the steps along the hull as (offer, extra cost, extra value), cheapest first. Each step has a lower
    value per extra dollar than the one before, which is what makes the greedy below match the LP relaxation."""
    opts = sorted(
        (float(c), float(v), j) for j, (v, c) in enumerate(zip(vals, costs, strict=True)) if np.isfinite(v) and v > 0
    )
    steps: list[tuple[int, float, float]] = []
    cur_c, cur_v = 0.0, 0.0
    while True:
        best, best_r = None, 0.0
        for c, v, j in opts:
            if v <= cur_v:
                continue
            dc = c - cur_c
            r = np.inf if dc <= 1e-12 else (v - cur_v) / dc
            if dc < -1e-12:
                continue
            if best is None or r > best_r or (r == best_r and v > best[1]):
                best, best_r = (c, v, j), r
        if best is None:
            return steps
        c, v, j = best
        steps.append((j, c - cur_c, v - cur_v))
        cur_c, cur_v = c, v


def greedy(values: np.ndarray, costs: np.ndarray, budget: float) -> np.ndarray:
    """Greedy by incremental value per dollar (the classic multiple-choice knapsack greedy).

    Each customer's options form a ladder: no offer, then upgrades along the convex hull of (cost, value).
    All upgrade steps across all customers are sorted by extra value per extra dollar and applied while they
    fit. A customer whose next step does not fit stops there, so they keep the best offer already bought."""
    n, _ = values.shape
    rows, extra_c, extra_v, offer = [], [], [], []
    for i in range(n):
        for j, dc, dv in _hull(values[i], costs[i]):
            rows.append(i)
            extra_c.append(dc)
            extra_v.append(dv)
            offer.append(j)
    if not rows:
        return np.full(n, NONE)
    rows_a, dc_a, dv_a, off_a = map(np.asarray, (rows, extra_c, extra_v, offer))
    ratio = dv_a / np.maximum(dc_a, 1e-9)
    order = np.lexsort((np.arange(len(rows_a)), -ratio))  # stable: a customer's steps stay in hull order
    assign = np.full(n, NONE)
    blocked = np.zeros(n, bool)
    spent = 0.0
    for s in order:
        i = rows_a[s]
        if blocked[i]:
            continue
        if spent + dc_a[s] > budget + 1e-9:
            blocked[i] = True
            continue
        assign[i] = off_a[s]
        spent += dc_a[s]
    return assign


def _lp_parts(values: np.ndarray, costs: np.ndarray):
    n, k = values.shape
    v = np.where(np.isfinite(values) & (values > 0), values, 0.0).ravel()
    one_per = csr_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), np.arange(n * k))), shape=(n, n * k))
    cost_row = csr_matrix(costs.ravel()[None, :])
    return v, vstack([one_per, cost_row]).tocsr()


def lp_bound(values: np.ndarray, costs: np.ndarray, budget: float) -> float | None:
    """Upper bound from the LP relaxation (fractional offers allowed). None when scipy is missing."""
    if not HAVE_SCIPY:
        return None
    n, _ = values.shape
    v, A = _lp_parts(values, costs)
    b = np.concatenate([np.ones(n), [budget]])
    res = linprog(-v, A_ub=A, b_ub=b, bounds=(0, 1), method="highs")
    return float(-res.fun) if res.status == 0 else None


def exact(values: np.ndarray, costs: np.ndarray, budget: float) -> np.ndarray:
    """Exact integer optimum with scipy milp. Fine for small and medium cases; used in tests and as a check."""
    n, k = values.shape
    v, A = _lp_parts(values, costs)
    b = np.concatenate([np.ones(n), [budget]])
    res = milp(-v, constraints=LinearConstraint(A, -np.inf, b), integrality=np.ones(n * k), bounds=(0, 1))
    x = np.round(res.x).reshape(n, k)
    assign = np.full(n, NONE)
    for i, j in zip(*np.nonzero(x > 0.5), strict=True):
        assign[i] = j
    return assign


def brute_force(values: np.ndarray, costs: np.ndarray, budget: float) -> tuple[float, np.ndarray]:
    """Enumerate every allocation. Only for tiny cases (n <= 8)."""
    n, k = values.shape
    best, best_a = 0.0, np.full(n, NONE)
    for combo in itertools.product(range(-1, k), repeat=n):
        a = np.array(combo)
        on = a != NONE
        c = costs[np.where(on)[0], a[on]].sum()
        if c > budget + 1e-9:
            continue
        val = values[np.where(on)[0], a[on]].sum()
        if np.isfinite(val) and val > best + 1e-12:
            best, best_a = float(val), a
    return best, best_a


def value_of(assign: np.ndarray, values: np.ndarray) -> float:
    idx = np.where(assign != NONE)[0]
    return float(values[idx, assign[idx]].sum())


def cost_of(assign: np.ndarray, costs: np.ndarray) -> float:
    idx = np.where(assign != NONE)[0]
    return float(costs[idx, assign[idx]].sum())


