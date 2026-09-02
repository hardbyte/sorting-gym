"""The ring environment, and what it does and does not change.

The ring keeps the pointers, the comparison bits and the instruction set, and
only removes the edges. It is a control for which invariant a policy depends
on, so these tests pin that it is solvable and that the array agents split on
it -- one survives, one does not.
"""

import numpy as np
import pytest

from sorting_gym.agents.scripted import bubble_sort_agent, ring_sort_agent
from sorting_gym.envs.ring import RingSortInterfaceEnv, descents, is_rotation_of_sorted

K = 3


@pytest.mark.parametrize("array,expected", [
    ([1, 2, 3], True), ([3, 1, 2], True), ([2, 3, 1], True),
    ([2, 1, 3], False), ([3, 2, 1], False),
    ([5, 5, 5], True), ([1], True), ([2, 1], True),
])
def test_is_rotation_of_sorted(array, expected):
    assert is_rotation_of_sorted(array) is expected


def test_the_ring_has_no_edges():
    """Bits 0 and 7 of every pointer row mark the array ends, and a ring has none."""
    env = RingSortInterfaceEnv(k=K)
    observation, _info = env.reset(seed=0)
    neighbours = observation['neighbour_view_comparisons']
    for i in range(K):
        assert neighbours[8 * i] == 0
        assert neighbours[8 * i + 7] == 0


def test_moves_and_swaps_wrap():
    env = RingSortInterfaceEnv(k=K)
    env.reset(seed=0)
    env.A = [1, 2, 3, 4]
    env.v[:] = [0, 0, 0]
    env.step((1, 0, False))                  # MoveVar(0, -1) off the front
    assert env.v[0] == 3, "moving left from the first element wraps to the last"
    env.v[:] = [3, 3, 3]
    env.step((0, 0))                         # SwapWithNext at the last element
    assert env.A == [4, 2, 3, 1], "swapping past the end wraps to the front"


def _run(agent, array, k=K):
    n = len(array)
    env = RingSortInterfaceEnv(k=k, max_episode_steps=40 * n * n)
    env.reset(seed=0)
    env.A = list(array)
    env.v[::2], env.v[1::2] = 0, n - 1
    env.steps_taken, env.episode_cost = 0, 0.0
    observation = env._get_obs()
    while True:
        observation, _r, terminated, truncated, _i = env.step(agent(observation, k))
        if terminated or truncated:
            return terminated, env.steps_taken, list(env.A)


@pytest.mark.parametrize("n", [5, 10, 20])
def test_ring_sort_agent_solves_the_ring(n):
    """Without this the environment would be untestable: a policy that fails on
    the ring would be indistinguishable from a ring that cannot be solved."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        array = list(rng.integers(0, 10, n))
        terminated, _steps, final = _run(ring_sort_agent, array)
        assert terminated, f"did not sort ring {array} within budget"
        assert is_rotation_of_sorted(final)
        assert sorted(final) == sorted(array), "elements must be preserved"


@pytest.mark.parametrize("n", [10, 20])
def test_the_array_bubble_agent_also_transfers(n):
    """It anchors on pointer comparisons rather than edge bits, so it survives
    losing the edges. The synthesized policy in examples/ does not."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        array = list(rng.integers(0, 10, n))
        terminated, _steps, final = _run(bubble_sort_agent, array)
        assert terminated and is_rotation_of_sorted(final)


def test_descents_counts_around_the_ring():
    assert descents([1, 2, 3]) == 1        # only the wrap from 3 back to 1
    assert descents([3, 2, 1]) == 2
    assert descents([5, 5, 5]) == 0
