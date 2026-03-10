import gymnasium
import sorting_gym
from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv

env: BasicNeuralSortInterfaceEnv = gymnasium.make('BasicNeuralSortInterfaceEnv-v0').unwrapped
observation, info = env.reset()

state, reward, terminated, truncated, info = env.step((0, 0))
env.render()
