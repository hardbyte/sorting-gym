import pytest
from gym.spaces import flatten

from sorting_gym.agents.scripted import bubble_sort_agent, insertion_sort_agent
from sorting_gym.envs.basic_neural_sort_interface import FunctionalNeuralSortInterfaceEnv
from tests.util import _test_sort_agent


def test_reset_gives_valid_observation():
    env = FunctionalNeuralSortInterfaceEnv(k=4)
    obs = flatten(env.nested_observation_space, env.reset())
    assert obs.shape[0] == 68


def test_bubble_sort_agent():
    """
    Functional environment should still work using the scripted
    Bubble Sort agent.
    """
    env = FunctionalNeuralSortInterfaceEnv(k=3)
    agent_f = bubble_sort_agent
    _test_sort_agent(agent_f, env, 100)


def test_bubble_sort_agent_not_enough_pointers():
    env = FunctionalNeuralSortInterfaceEnv(k=2)
    agent_f = bubble_sort_agent
    with pytest.raises(IndexError):
        _test_sort_agent(agent_f, env, 100)

