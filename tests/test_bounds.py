"""Optimal values and bounds for the combinatorial environments.

These make a score interpretable: without a ceiling, "the policy collected 47"
says nothing about whether it did well.
"""

import itertools

import numpy as np
import pytest

from sorting_gym.bounds import (
    approximation_ratio, bin_packing_lower_bound, bin_packing_optimal,
    knapsack_lp_bound, knapsack_optimal)
from sorting_gym.envs.bin_packing import BinPackingEnv
from sorting_gym.envs.knapsack import KnapsackEnv


def _brute_force_knapsack(items, capacity):
    best = 0
    for chosen in itertools.product([0, 1], repeat=len(items)):
        weight = sum(w for (w, _v), take in zip(items, chosen) if take)
        if weight <= capacity:
            best = max(best, sum(v for (_w, v), take in zip(items, chosen) if take))
    return best


@pytest.mark.parametrize("seed", range(15))
def test_knapsack_dp_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 11))
    items = [(int(rng.integers(1, 20)), int(rng.integers(1, 40))) for _ in range(n)]
    capacity = int(rng.integers(1, 60))
    assert knapsack_optimal(items, capacity) == _brute_force_knapsack(items, capacity)


@pytest.mark.parametrize("seed", range(10))
def test_lp_bound_is_an_upper_bound(seed):
    """The fractional relaxation can never be below the 0/1 optimum."""
    rng = np.random.default_rng(seed)
    items = [(int(rng.integers(1, 20)), int(rng.integers(1, 40)))
             for _ in range(int(rng.integers(1, 9)))]
    capacity = int(rng.integers(1, 50))
    assert knapsack_lp_bound(items, capacity) >= knapsack_optimal(items, capacity) - 1e-9


def test_knapsack_edge_cases():
    assert knapsack_optimal([], 10) == 0
    assert knapsack_optimal([(20, 5)], 10) == 0, "an item heavier than the sack"
    assert knapsack_optimal([(5, 3), (5, 4)], 10) == 7


@pytest.mark.parametrize("sizes,capacity,expected", [
    ([4, 8, 1, 4, 2, 1], 10, 2),
    ([5, 5, 5], 5, 3),
    ([1, 1, 1, 1], 4, 1),
    ([], 10, 0),
    ([7, 3, 5, 5], 10, 2),
])
def test_bin_packing_optimal(sizes, capacity, expected):
    assert bin_packing_optimal(sizes, capacity) == expected


@pytest.mark.parametrize("seed", range(10))
def test_bin_packing_optimal_is_never_below_the_lower_bound(seed):
    rng = np.random.default_rng(seed)
    capacity = int(rng.integers(10, 30))
    sizes = [int(rng.integers(1, capacity + 1)) for _ in range(int(rng.integers(1, 9)))]
    assert bin_packing_optimal(sizes, capacity) >= bin_packing_lower_bound(sizes, capacity)


def test_bin_packing_refuses_instances_it_cannot_finish():
    with pytest.raises(ValueError, match="exceeds"):
        bin_packing_optimal(list(range(1, 40)), 100)


def test_bin_packing_rejects_an_oversized_item():
    with pytest.raises(ValueError, match="larger than a bin"):
        bin_packing_optimal([5, 20], 10)


@pytest.mark.parametrize("achieved,optimal,sense,expected", [
    (8, 10, "max", 0.8), (10, 10, "max", 1.0),
    (2, 2, "min", 1.0), (4, 2, "min", 0.5),
])
def test_approximation_ratio(achieved, optimal, sense, expected):
    assert approximation_ratio(achieved, optimal, sense) == pytest.approx(expected)


def test_knapsack_env_reports_the_ratio_on_termination():
    env = KnapsackEnv(k=4, starting_min_items=5)
    env.reset(seed=0)
    _obs, _r, terminated, _tr, info = env.step((4, 0))       # Finish immediately
    assert terminated
    assert info["optimal_value"] == env.optimal_value()
    assert info["approximation_ratio"] == 0.0, "selecting nothing achieves nothing"


def test_bin_packing_env_reports_bounds_on_termination():
    env = BinPackingEnv(k=4, starting_min_items=5)
    env.reset(seed=0)
    # Give every item its own bin: correct, and about as bad as packing gets.
    terminated = False
    for _ in range(env.num_items):
        _obs, _r, terminated, _tr, info = env.step((1, 0))   # AssignToNewBin(v0)
        if terminated:
            break
        _obs, _r, terminated, _tr, info = env.step((2, 0, True))   # MoveVar(v0, +1)
        if terminated:
            break
    assert terminated
    assert info["lower_bound_bins"] <= info["num_bins"]
    assert 0 < info["approximation_ratio"] <= 1.0
