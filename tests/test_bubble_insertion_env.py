import pytest
import numpy as np
from gym.spaces import flatten, flatdim

from sorting_gym.agents.scripted import bubble_sort_agent, insertion_sort_agent
from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv


def test_observation_size():
    env = BasicNeuralSortInterfaceEnv(k=4)
    assert flatdim(env.observation_space) == 68


def test_reset_gives_valid_observation():
    env = BasicNeuralSortInterfaceEnv(k=4)
    obs = flatten(env.nested_observation_space, env.reset())
    assert obs.shape[0] == 68


def _test_sort_agent(agent_f, env, number_of_problems=1000, max_steps = 1000, verbose=False):
    k = env.k
    for problem in range(1, number_of_problems):
        obs = env.reset()
        for step in range(1, max_steps+1):
            action = agent_f(obs, k)
            if verbose: print("Action: ", action)
            obs, reward, is_done, info = env.step(action)
            if verbose: print(info)
            if is_done:
                if verbose:
                    print(f"Solved problem {problem} of size {len(env.A)} in {step} steps")
                break
            if step == max_steps:
                pytest.fail(f"Didn't solve problem {problem} of size {len(env.A)}.")


def test_bubble_sort_agent():
    """
    Tests the environment using a Bubble Sort agent.

    c.f. Algorithm 2 - pg 19
    """
    env = BasicNeuralSortInterfaceEnv(k=3)
    agent_f = bubble_sort_agent
    _test_sort_agent(agent_f, env, 1000)


def test_bubble_sort_agent_not_enough_pointers():
    env = BasicNeuralSortInterfaceEnv(k=2)
    agent_f = bubble_sort_agent
    with pytest.raises(IndexError):
        _test_sort_agent(agent_f, env, 1000)


def test_insertion_sort_agent():
    """
    Tests the environment using an Insertion Sort agent.

    c.f. Algorithm 4 - pg 20
    """
    k = 3
    env = BasicNeuralSortInterfaceEnv(k=k)
    _test_sort_agent(insertion_sort_agent, env, verbose=False)


if __name__ == '__main__':
    #test_bubble_sort_agent()
    test_insertion_sort_agent()