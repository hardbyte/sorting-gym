import pytest
from gym.spaces import flatten, flatdim

from sorting_gym.agents.scripted import bubble_sort_agent, insertion_sort_agent
from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv
from tests.util import _test_sort_agent


def test_observation_size():
    env = BasicNeuralSortInterfaceEnv(k=4)
    assert flatdim(env.observation_space) == 68


def test_reset_gives_valid_observation():
    env = BasicNeuralSortInterfaceEnv(k=4)
    obs = flatten(env.nested_observation_space, env.reset())
    assert obs.shape[0] == 68


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
