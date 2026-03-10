from gymnasium.envs.registration import register
from sorting_gym.parametric_space import DiscreteParametric

register(
    id='SortTapeAlgorithmicEnv-v0',
    entry_point='sorting_gym.envs.tape:SortTapeAlgorithmicEnv'
)

register(
    id='BasicNeuralSortInterfaceEnv-v0',
    entry_point='sorting_gym.envs.basic_neural_sort_interface:BasicNeuralSortInterfaceEnv'
)

register(
    id='FunctionalNeuralSortInterfaceEnv-v0',
    entry_point='sorting_gym.envs.functional_neural_sort_interface:FunctionalNeuralSortInterfaceEnv'
)

register(
    id='KnapsackEnv-v0',
    entry_point='sorting_gym.envs.knapsack:KnapsackEnv'
)

register(
    id='BinPackingEnv-v0',
    entry_point='sorting_gym.envs.bin_packing:BinPackingEnv'
)

register(
    id='JobShopSchedulingEnv-v0',
    entry_point='sorting_gym.envs.job_shop:JobShopSchedulingEnv'
)

