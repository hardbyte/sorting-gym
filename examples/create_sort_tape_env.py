import gymnasium
from sorting_gym.envs.tape import SortTapeAlgorithmicEnv

env: SortTapeAlgorithmicEnv = gymnasium.make('SortTapeAlgorithmicEnv-v0').unwrapped
observation, info = env.reset()

print(f"Input data: {env.input_data}")
print(f"Target: {env.target}")
