import gym
import sorting_gym
from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv

env: BasicNeuralSortInterfaceEnv = gym.make('BasicNeuralSortInterfaceEnv-v0')
observation = env.reset()

state, reward, done, info = env.step((0, 0))
env.render()
