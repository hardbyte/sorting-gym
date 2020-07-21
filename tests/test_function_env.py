import pytest
from gym.spaces import flatten

from sorting_gym.agents.scripted import bubble_sort_agent, insertion_sort_agent
from sorting_gym.envs.functional_neural_sort_interface import FunctionalNeuralSortInterfaceEnv
from tests.util import _test_sort_agent


def test_reset_gives_valid_observation():
    env = FunctionalNeuralSortInterfaceEnv(k=4, number_of_functions=5)
    obs = flatten(env.nested_observation_space, env.reset())
    assert obs.shape[0] == 68 + 5


def test_function_env_preserves_function_id():
    """
    Create a functional environment with 2 functions taking 0 args and returning 0 args
    """
    env = FunctionalNeuralSortInterfaceEnv(k=3, number_of_functions=2, function_inputs=0, function_returns=0)
    original_obs = env.reset()
    assert original_obs['current_function'] == -1
    obs, reward, done, info = env.step((3, 0))
    assert obs['current_function'] == 0
    obs, reward, done, info = env.step((3, 1))
    assert obs['current_function'] == 1
    # return
    obs, reward, done, info = env.step((4,))
    assert obs['current_function'] == 0
    obs, reward, done, info = env.step((4,))
    assert obs['current_function'] == -1


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

