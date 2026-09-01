"""Per-instruction costs, and what they can and cannot change.

Costs are the objective the agent optimises, so they decide which algorithm is
best rather than merely rescaling the reward.
"""

import numpy as np
import pytest

from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv
from sorting_gym.agents.scripted import bubble_sort_agent, insertion_sort_agent

UNIFORM = {"SwapWithNext": 1, "MoveVar": 1, "AssignVar": 1}
EXPENSIVE_SWAP = {"SwapWithNext": 20, "MoveVar": 1, "AssignVar": 1}
EXPENSIVE_ASSIGN = {"SwapWithNext": 1, "MoveVar": 1, "AssignVar": 20}


def _episode_cost(agent, costs, n=20, trials=8, k=3):
    rng = np.random.default_rng(0)
    totals = []
    for _ in range(trials):
        array = list(rng.integers(0, 10, n))
        while array == sorted(array):
            array = list(rng.integers(0, 10, n))
        env = BasicNeuralSortInterfaceEnv(
            k=k, max_episode_steps=20 * n * n, instruction_costs=costs)
        env.reset(seed=0)
        env.A, env.tape_env.target = list(array), sorted(array)
        env.v[::2], env.v[1::2] = 0, n - 1
        env.steps_taken, env.episode_cost = 0, 0.0
        observation = env._get_obs()
        while True:
            observation, _r, terminated, truncated, _i = env.step(agent(observation, k))
            if terminated or truncated:
                break
        totals.append(env.episode_cost)
    return float(np.mean(totals))


def test_default_costs_keep_the_original_reward():
    env = BasicNeuralSortInterfaceEnv(k=3)
    env.reset(seed=0)
    _obs, reward, _t, _tr, info = env.step((1, 0, True))
    assert reward == -1
    assert info["cost"] == 1.0


def test_cost_is_charged_per_instruction():
    env = BasicNeuralSortInterfaceEnv(k=3, instruction_costs=EXPENSIVE_ASSIGN)
    env.reset(seed=0)
    _obs, reward, _t, _tr, info = env.step((2, 0, 1))
    assert reward == -20
    assert info["episode_cost"] == 20


def test_unknown_instruction_name_is_rejected():
    with pytest.raises(ValueError, match="unknown instruction"):
        BasicNeuralSortInterfaceEnv(k=3, instruction_costs={"Nope": 2})


def test_swap_cost_cannot_change_which_algorithm_wins():
    """SwapWithNext is adjacent-only, so every correct policy swaps once per
    inversion. Pricing swaps shifts both totals by the same amount."""
    bubble_delta = (_episode_cost(bubble_sort_agent, EXPENSIVE_SWAP)
                    - _episode_cost(bubble_sort_agent, UNIFORM))
    insertion_delta = (_episode_cost(insertion_sort_agent, EXPENSIVE_SWAP)
                       - _episode_cost(insertion_sort_agent, UNIFORM))
    assert bubble_delta == pytest.approx(insertion_delta)


def test_the_cheaper_algorithm_depends_on_the_cost_model():
    """The point of costs: the objective decides which algorithm is best."""
    assert (_episode_cost(insertion_sort_agent, UNIFORM)
            < _episode_cost(bubble_sort_agent, UNIFORM))
    assert (_episode_cost(bubble_sort_agent, EXPENSIVE_ASSIGN)
            < _episode_cost(insertion_sort_agent, EXPENSIVE_ASSIGN))
