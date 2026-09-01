"""Deterministic edge cases for the sorting agents.

The random instance stream deliberately excludes already sorted arrays (they
terminate on whatever instruction the agent happens to pick first, so they
measure almost nothing while distorting the curriculum). The degenerate cases
are still worth covering, so they are pinned here explicitly instead.
"""

import pytest

from sorting_gym.agents.scripted import bubble_sort_agent, insertion_sort_agent
from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv
from sorting_gym.text import text_bubble_sort_agent, text_insertion_sort_agent

K = 3

AGENTS = [
    ("bubble", bubble_sort_agent),
    ("insertion", insertion_sort_agent),
    ("text_bubble", text_bubble_sort_agent),
    ("text_insertion", text_insertion_sort_agent),
]

INSTANCES = {
    "sorted": [0, 1, 2, 3, 4, 5, 6, 7],
    "reverse_sorted": [7, 6, 5, 4, 3, 2, 1, 0],
    "all_equal": [5] * 8,
    "near_sorted": [0, 1, 2, 3, 4, 6, 5, 7],
    "sorted_with_duplicates": [1, 1, 2, 2, 3, 3],
    "reverse_with_duplicates": [3, 3, 2, 2, 1, 1],
    "single_element": [4],
    "two_sorted": [1, 2],
    "two_unsorted": [2, 1],
    "one_out_of_place": [1, 2, 3, 4, 5, 6, 7, 0],
}


def _install(env, array):
    """Put a chosen instance into the environment, mirroring reset's bookkeeping."""
    env.A = list(array)
    env.tape_env.target = sorted(array)
    env.v[::2] = 0
    env.v[1::2] = len(array) - 1
    env.steps_taken = 0
    return env._get_obs()


def _run(agent, array):
    env = BasicNeuralSortInterfaceEnv(k=K)
    env.reset(seed=0)
    obs = _install(env, array)
    while True:
        obs, reward, terminated, truncated, info = env.step(agent(obs, K))
        if terminated or truncated:
            return terminated, env.steps_taken, list(env.A)


@pytest.mark.parametrize("name", sorted(INSTANCES))
@pytest.mark.parametrize("label,agent", AGENTS, ids=[a[0] for a in AGENTS])
def test_agent_sorts_edge_case(name, label, agent):
    array = INSTANCES[name]
    terminated, steps, final = _run(agent, array)
    assert terminated, f"{label} failed to sort {name} within budget (got {final})"
    assert final == sorted(array)


@pytest.mark.parametrize("label,agent", AGENTS, ids=[a[0] for a in AGENTS])
def test_sorted_input_is_not_disturbed(label, agent):
    """The one property a pre-sorted instance genuinely tests.

    A spurious SwapWithNext would break sorted order and force the agent to
    recover, so terminating in a single step is what "did no damage" looks like.
    """
    array = INSTANCES["sorted"]
    terminated, steps, final = _run(agent, array)
    assert terminated and steps == 1 and final == array


def test_allow_sorted_instances_samples_the_natural_distribution():
    """Evaluation may want the untouched distribution, sorted instances included."""
    env = BasicNeuralSortInterfaceEnv(k=K, allow_sorted_instances=True)
    seen_sorted = False
    for _ in range(5000):
        env.reset()
        if env.A == sorted(env.A):
            seen_sorted = True
            break
    assert seen_sorted, "expected at least one sorted instance when they are allowed"
