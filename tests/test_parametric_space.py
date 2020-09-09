from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv
from sorting_gym.envs.functional_neural_sort_interface import FunctionalNeuralSortInterfaceEnv


def test_parametric_space():
    env = BasicNeuralSortInterfaceEnv(k=2)
    for i in range(100):
        sample = env.action_space.sample()
        assert env.action_space.contains(sample)


def test_parametric_space_2():
    env = BasicNeuralSortInterfaceEnv(k=4)
    for i in range(100):
        sample = env.action_space.sample()
        assert env.action_space.contains(sample)


def test_parametric_space_3():
    env = FunctionalNeuralSortInterfaceEnv(k=3, number_of_functions=2, function_inputs=2, function_returns=1)
    for i in range(100):
        sample = env.action_space.sample()
        assert env.action_space.contains(sample)
