from gym.envs.registration import register
from sorting_gym.parametric_space import DiscreteParametric

register(
    id='SortTapeAlgorithmicEnv-v0',
    entry_point='sorting_gym.envs.tape:SortTapeAlgorithmicEnv'
)

register(
    id='BasicNeuralSortInterfaceEnv-v0',
    entry_point='sorting_gym.envs.bubblesort:BasicNeuralSortInterfaceEnv'
)

