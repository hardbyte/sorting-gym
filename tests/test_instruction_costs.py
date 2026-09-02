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


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), float("-inf")])
def test_costs_must_be_finite_and_non_negative(bad):
    """A negative price pays the agent to act: looping on a bounded no-op such
    as MoveVar at an edge would earn reward until truncation. NaN spreads into
    every reward and comparison downstream."""
    with pytest.raises(ValueError, match="finite and non-negative"):
        BasicNeuralSortInterfaceEnv(k=3, instruction_costs={"MoveVar": bad})


def test_zero_cost_is_allowed():
    env = BasicNeuralSortInterfaceEnv(k=3, instruction_costs={"MoveVar": 0})
    env.reset(seed=0)
    _obs, reward, _t, _tr, info = env.step((1, 0, True))
    assert reward == 0 and info["cost"] == 0


def test_a_swap_at_the_right_edge_is_a_charged_no_op():
    """`op_swap_with_next` clamps to the last index, so a swap there exchanges
    an element with itself but is still billed."""
    env = BasicNeuralSortInterfaceEnv(k=3)
    env.reset(seed=0)
    env.A, env.tape_env.target = [3, 1, 2], [1, 2, 3]
    env.v[:] = [2, 2, 2]
    _obs, reward, _t, _tr, info = env.step((0, 0))
    assert env.A == [3, 1, 2]
    assert reward == -1 and info["cost"] == 1


def test_swap_count_is_not_invariant_across_correct_policies():
    """Only swap-minimal policies spend one swap per inversion.

    At reset v1 is at the right edge, so a policy may prepend a charged no-op
    swap there and still sort. The swap count therefore is not fixed across all
    correct policies, and an expensive-swap model can separate a wasteful policy
    from a swap-minimal one - it just cannot separate bubble from insertion,
    since both are swap-minimal.
    """
    def wasteful(observation, k, seen=[]):
        if not seen:
            seen.append(True)
            return (0, 1)          # SwapWithNext(v1) at the right edge: a no-op
        return bubble_sort_agent(observation, k)

    plain = _episode_cost(bubble_sort_agent, EXPENSIVE_SWAP, n=8, trials=1)
    padded = _episode_cost(wasteful, EXPENSIVE_SWAP, n=8, trials=1)
    assert padded == plain + EXPENSIVE_SWAP["SwapWithNext"]
