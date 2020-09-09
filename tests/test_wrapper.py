import numpy as np
import pytest
from gym.spaces import Box, flatdim, MultiDiscrete, Discrete, Tuple, MultiBinary

from sorting_gym import DiscreteParametric
from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv
from sorting_gym.envs.wrappers import BoxActionSpaceWrapper, MultiDiscreteActionSpaceWrapper, \
    DisjointMultiDiscreteActionSpaceWrapper, merge_discrete_spaces


def test_parametric_multi_discrete_wrap():
    for k in [4, 5, 6]:
        env = BasicNeuralSortInterfaceEnv(k=k)
        assert isinstance(env.action_space, DiscreteParametric)
        assert isinstance(MultiDiscreteActionSpaceWrapper(env).action_space, MultiDiscrete)
        assert MultiDiscreteActionSpaceWrapper(env).action_space.shape[0] == 6


def test_parametric_multi_discrete_wrap_actions():
    env = MultiDiscreteActionSpaceWrapper(BasicNeuralSortInterfaceEnv(k=2))
    for _ in range(100):
        random_action = env.action_space.sample()
        assert env.action_space.contains(random_action)
        env.step(random_action)


def test_parametric_flat_wrap():
    k = 4
    env = BasicNeuralSortInterfaceEnv(k=k)
    assert flatdim(env.observation_space) == 68
    assert isinstance(BoxActionSpaceWrapper(env).action_space, Box)
    num_instructions = 3
    num_args = 4 * k + 1
    assert BoxActionSpaceWrapper(env).action_space.shape[0] == num_instructions + num_args


def test_parametric_flat_wrap_actions():
    k = 2
    env = BoxActionSpaceWrapper(BasicNeuralSortInterfaceEnv(k=k))
    random_action = env.action_space.sample()

    # The disjoint action space for action (1) is a Tuple(Discrete(2), MultiBinary(1))
    # Have to skip over the 3 parameter space bits, and 2 one hot encoding bits for action (0)
    # then bits 5 and 6 correspond to the Discrete(2), and bit 7 to the multibinary(1).
    encoded_bits = [
        [(1, 0, 0), (0, 0)],
        [(0, 1, 0), (1, 0)],
        [(0, 1, 1), (1, 1)],
    ]

    for bits_to_set, expected_results in encoded_bits:
        action = np.zeros_like(random_action)
        action[1] = 1.0
        action[5:8] = bits_to_set
        result = env.action(action)

        assert result[0] == 1
        assert result[1] == expected_results[0]
        assert result[2][0] == pytest.approx(expected_results[1])


def test_parametric_flat_wrap_invalid_actions():
    # Test where the discrete args are all zero
    k = 2
    env = BoxActionSpaceWrapper(BasicNeuralSortInterfaceEnv(k=k))
    random_action = env.action_space.sample()
    action = np.zeros_like(random_action)
    action[1] = 1.0
    with pytest.raises(ValueError):
        env.action(action)


def test_parametric_wrapped_action_step():
    k = 2
    env = BoxActionSpaceWrapper(BasicNeuralSortInterfaceEnv(k=k))
    # A random sample of the wrapped space won't be valid
    action = np.zeros_like(env.action_space.sample())
    # Set action to be SwapWithNext(1)
    action[0] = 1.0
    action[4] = 1.0

    env.step(action)


def test_parametric_wrapped_action_samples():
    k = 2
    env = BoxActionSpaceWrapper(BasicNeuralSortInterfaceEnv(k=k))
    # A random sample of the wrapped space via this method should always be valid
    for _ in range(100):
        action = env.action_space_sample()
        env.step(action)


@pytest.mark.xfail
def test_disjoint_parametric_wrapped_action_samples():
    k = 2
    env = DisjointMultiDiscreteActionSpaceWrapper(BasicNeuralSortInterfaceEnv(k=k))

    # A random sample of the wrapped space via this method should always be valid
    for _ in range(100):
        action = env.action_space.sample()
        env.step(action)


def test_merge_discrete_space():
    k = 4
    merged_discrete_space = merge_discrete_spaces([Discrete(k)])

    assert isinstance(merged_discrete_space, MultiDiscrete)
    for i in range(k):
        assert merged_discrete_space.contains(np.array([i], dtype=np.int64))
    assert not merged_discrete_space.contains(np.array([k], dtype=np.int64))


def test_merge_discrete_spaces():
    k = 4
    space_to_merge = [
        Tuple([Discrete(k), Discrete(k)]),
    ]

    merged_discrete_space = merge_discrete_spaces(space_to_merge)

    assert isinstance(merged_discrete_space, MultiDiscrete)
    assert merged_discrete_space.contains(np.array([0, 0], dtype=np.int64))
    assert merged_discrete_space.contains(np.array([0, 1], dtype=np.int64))
    assert merged_discrete_space.contains(np.array([2, 2], dtype=np.int64))
    assert merged_discrete_space.contains(np.array([3, 3], dtype=np.int64))
    assert not merged_discrete_space.contains(np.array([4, 4], dtype=np.int64))


def test_merge_discrete_spaces_flat():
    k = 4
    space_to_merge = [
        Discrete(k), Discrete(k),
    ]

    merged_discrete_space = merge_discrete_spaces(space_to_merge)

    assert isinstance(merged_discrete_space, MultiDiscrete)
    assert merged_discrete_space.contains(np.array([0, 0], dtype=np.int64))
    assert merged_discrete_space.contains(np.array([0, 1], dtype=np.int64))
    assert merged_discrete_space.contains(np.array([2, 2], dtype=np.int64))
    assert merged_discrete_space.contains(np.array([3, 3], dtype=np.int64))
    assert not merged_discrete_space.contains(np.array([4, 4], dtype=np.int64))


def test_merge_discrete_and_multibinary():
    space_to_merge = [Tuple([Discrete(4), MultiBinary(1)])]
    merged_discrete_space = merge_discrete_spaces(space_to_merge)
    assert isinstance(merged_discrete_space, MultiDiscrete)
    assert merged_discrete_space.contains(np.array([0, 0], dtype=np.int64))
    assert merged_discrete_space.contains(np.array([3, 1], dtype=np.int64))
    assert not merged_discrete_space.contains(np.array([3, 2], dtype=np.int64))
    assert not merged_discrete_space.contains(np.array([4, 1], dtype=np.int64))

    merged_discrete_space.sample()