import numpy as np
import pytest

from uplift import budget as bud


def random_case(seed, n=6, k=3):
    rng = np.random.default_rng(seed)
    costs = rng.uniform(0.5, 10, (n, k))
    values = rng.normal(5, 6, (n, k))
    values[rng.random((n, k)) < 0.15] = -np.inf
    return values, costs, float(costs.sum() / 4)


@pytest.mark.parametrize("seed", range(5))
def test_exact_matches_brute_force(seed):
    v, c, b = random_case(seed)
    best, _ = bud.brute_force(v, c, b)
    assert bud.value_of(bud.exact(v, c, b), v) == pytest.approx(best)


@pytest.mark.parametrize("seed", range(5))
def test_greedy_feasible_and_bounded(seed):
    v, c, b = random_case(seed)
    a = bud.greedy(v, c, b)
    best, _ = bud.brute_force(v, c, b)
    assert bud.cost_of(a, c) <= b + 1e-9
    assert bud.value_of(a, v) <= best + 1e-9
    assert best <= bud.lp_bound(v, c, b) + 1e-6


def test_greedy_optimal_on_small_case():
    # Three customers, two offers each, budget 12.
    values = np.array([[6.0, 9.0], [4.0, 4.5], [3.0, -1.0]])
    costs = np.array([[2.0, 8.0], [2.0, 6.0], [3.0, 1.0]])
    best, best_a = bud.brute_force(values, costs, 12)
    a = bud.greedy(values, costs, 12)
    assert bud.value_of(a, values) == pytest.approx(best)
    assert best == pytest.approx(13.5)
    assert a.tolist() == best_a.tolist()


def test_incremental_greedy_avoids_cheap_offer_trap():
    # A near-free offer with a big value per dollar would lock the customer out of the much better one.
    values = np.array([[5.0, 30.0]])
    costs = np.array([[1.0, 10.0]])
    assert bud.value_of(bud.greedy_naive(values, costs, 10), values) == 5.0
    assert bud.value_of(bud.greedy(values, costs, 10), values) == 30.0


@pytest.mark.parametrize("budget", [0, 5, 50, 500])
def test_budget_respected(budget):
    v, c, _ = random_case(3, n=40)
    a = bud.greedy(v, c, budget)
    assert bud.cost_of(a, c) <= budget + 1e-9


def test_never_assigns_negative_or_ineligible():
    v = np.array([[-1.0, -np.inf], [-5.0, -2.0]])
    assert (bud.greedy(v, np.ones((2, 2)), 100) == -1).all()


def test_greedy_close_to_lp_bound_at_scale():
    rng = np.random.default_rng(0)
    n = 3000
    c = rng.uniform(1, 20, (n, 3))
    v = rng.normal(2, 8, (n, 3))
    b = 5000.0
    a = bud.greedy(v, c, b)
    assert bud.value_of(a, v) >= 0.99 * bud.lp_bound(v, c, b)


def test_hull_steps_have_falling_ratios():
    steps = bud._hull(np.array([5.0, 30.0, 31.0]), np.array([1.0, 10.0, 30.0]))
    ratios = [dv / dc for _, dc, dv in steps]
    assert [s[0] for s in steps] == [0, 1, 2]
    assert ratios == sorted(ratios, reverse=True)
