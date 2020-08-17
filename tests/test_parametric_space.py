from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv
from sorting_gym.envs.functional_neural_sort_interface import FunctionalNeuralSortInterfaceEnv


def test_parametric_space():
    env = BasicNeuralSortInterfaceEnv(k=2)
    for i in range(100):
        sample = env.action_space.sample()
        assert env.action_space.contains(sample)


def test_parametric_space_2():
    env = FunctionalNeuralSortInterfaceEnv(k=3, number_of_functions=2, function_inputs=0, function_returns=0)
    for i in range(100):
        sample = env.action_space.sample()
        assert env.action_space.contains(sample)
