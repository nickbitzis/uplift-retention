"""Plans, offers, and the customer lifetime value model.

Everything money related lives here so the generator, the policy math, and the API agree on it.

CLV model (deliberately simple, and the same one the ground truth uses):
    monthly margin   = 50% of the plan price (the rest covers content licensing, streaming, and payment fees)
    remaining months = 12 + 0.4 x tenure in months, capped at 36 (customers who stayed long tend to stay long)
    CLV              = monthly margin x remaining months
A customer retained on a cheaper plan (the downgrade offer) is worth the CLV at the cheaper price.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PLANS = ("basic", "standard", "premium")
PRICES = {"basic": 8.99, "standard": 15.49, "premium": 22.99}
DOWNGRADE_TO = {"standard": "basic", "premium": "standard"}
MARGIN = 0.50
CONTACT_COST = 0.50  # email, push, and the in-app banner, per targeted customer


@dataclass(frozen=True)
class Offer:
    key: str
    label: str
    description: str


OFFERS: tuple[Offer, ...] = (
    Offer("discount", "25% off for 3 months", "Applied as an account credit when sent."),
    Offer("free_month", "One free month", "The next bill is waived."),
    Offer("downgrade", "Plan downgrade", "Move to the next cheaper plan with no fee. Not offered on Basic."),
)
OFFER_KEYS = tuple(o.key for o in OFFERS)


def price_array(plan: np.ndarray) -> np.ndarray:
    return np.vectorize(PRICES.get, otypes=[float])(plan)


def remaining_months(tenure: np.ndarray) -> np.ndarray:
    return np.minimum(12.0 + 0.4 * np.asarray(tenure, dtype=float), 36.0)


def clv(price: np.ndarray, tenure: np.ndarray) -> np.ndarray:
    return MARGIN * np.asarray(price, dtype=float) * remaining_months(tenure)


def downgrade_price(plan: np.ndarray) -> np.ndarray:
    """Price after a downgrade. Basic has no cheaper plan, so it returns the same price (and is ineligible)."""
    return np.array([PRICES[DOWNGRADE_TO.get(p, p)] for p in plan], dtype=float)


def offer_economics(plan: np.ndarray, tenure: np.ndarray) -> dict[str, np.ndarray]:
    """Per customer, per offer: cost when sent, CLV if retained under that offer, and eligibility.

    Returns arrays shaped (n, 3) in OFFER_KEYS order plus `clv0`, the CLV if retained with no offer.
    """
    plan = np.asarray(plan)
    price = price_array(plan)
    base = clv(price, tenure)
    down = downgrade_price(plan)
    cost = np.column_stack([0.25 * 3 * price, price, np.zeros_like(price)]) + CONTACT_COST
    clv_k = np.column_stack([base, base, clv(down, tenure)])
    eligible = np.column_stack([np.ones(len(plan), bool), np.ones(len(plan), bool), plan != "basic"])
    return {"cost": cost, "clv_k": clv_k, "clv0": base, "eligible": eligible}
