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
    action = env.action_space.sample()

    """
    First 3 values represent the instruction, could be real:
    [0.5035229 , 0.7449214 , 0.58745563]
    
    """