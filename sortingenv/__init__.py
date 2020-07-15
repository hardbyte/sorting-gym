from gym.envs.registration import register


register(
    id='SortTapeAlgorithmicEnv-v0',
    entry_point='sortingenv.envs.tape:SortTapeAlgorithmicEnv'
)
