import numpy as np
from gym.spaces import Box, flatdim

from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv
from sorting_gym.envs.wrappers import SimpleActionSpace


def test_parametric_flat_wrap():
    k = 4
    env = BasicNeuralSortInterfaceEnv(k=k)
    assert flatdim(env.observation_space) == 68
    assert isinstance(SimpleActionSpace(env).action_space, Box)
    num_instructions = 3
    num_args = 4 * k + 1
    assert SimpleActionSpace(env).action_space.shape[0] == num_instructions + num_args


def test_parametric_flat_wrap_actions():
    k = 2
    env = SimpleActionSpace(BasicNeuralSortInterfaceEnv(k=k))
    random_action = env.action_space.sample()
    action = np.zeros_like(random_action)

    # The disjoint action space for (1) is a Discrete(2), MultiBinary(1)
    action[1] = 1.0
    action[4] = 1.0
    transformed_action = env.action(action)
    assert transformed_action == (1, np.array(0, 0))


def test_parametric_wrapped_action_step():
    k = 2
    env = SimpleActionSpace(BasicNeuralSortInterfaceEnv(k=k))
    # A random sample of the wrapped space won't be valid
    action = np.zeros_like(env.action_space.sample())
    # Set action to be SwapWithNext(1)
    action[0] = 1.0
    action[4] = 1.0

    env.step(action)


