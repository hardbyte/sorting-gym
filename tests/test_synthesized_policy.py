"""The language model's program must keep sorting.

Also an end-to-end check of the synthesis path: the facts DSL, the instruction
grammar, and the environment all have to line up for this to pass.
"""

import importlib.util
import pathlib

import pytest

from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv
from sorting_gym.text import parse_action, parse_observation, render_observation

POLICY_PATH = pathlib.Path(__file__).parent.parent / "examples" / "synthesized" / "bubble_policy.py"


def _load_policy():
    spec = importlib.util.spec_from_file_location("synthesized_policy", POLICY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.policy


def _run(policy, array, k=3):
    env = BasicNeuralSortInterfaceEnv(k=k, max_episode_steps=8 * len(array) ** 2)
    env.reset(seed=0)
    env.A, env.tape_env.target = list(array), sorted(array)
    env.v[::2], env.v[1::2] = 0, len(array) - 1
    env.steps_taken = 0
    observation = env._get_obs()
    while True:
        facts = parse_observation(render_observation(observation, k), k)
        observation, _r, terminated, truncated, _i = env.step(parse_action(policy(facts), k))
        if terminated or truncated:
            return terminated, env.steps_taken, list(env.A)


@pytest.mark.parametrize("n", [5, 10, 20, 40])
def test_synthesized_policy_sorts(n):
    """Includes lengths beyond the 5/10/20 it was scored on."""
    import numpy as np
    policy = _load_policy()
    rng = np.random.default_rng(0)
    for _ in range(3):
        array = list(rng.integers(0, 10, n))
        terminated, steps, final = _run(policy, array)
        assert terminated, f"did not sort {array} within budget (reached {final})"
        assert final == sorted(array)


@pytest.mark.parametrize("array", [
    [0, 1, 2, 3], [3, 2, 1, 0], [5, 5, 5, 5], [1], [2, 1],
])
def test_synthesized_policy_on_edge_cases(array):
    policy = _load_policy()
    terminated, _steps, final = _run(policy, array)
    assert terminated and final == sorted(array)
