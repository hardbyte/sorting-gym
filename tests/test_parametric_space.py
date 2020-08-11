from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv


def test_parametric_space():
    env = BasicNeuralSortInterfaceEnv(k=2)
    for i in range(100):
        sample = env.action_space.sample()
        assert env.action_space.contains(sample)