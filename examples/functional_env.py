
import gym
import sorting_gym
from sorting_gym.envs.functional_neural_sort_interface import FunctionalNeuralSortInterfaceEnv

env = FunctionalNeuralSortInterfaceEnv()
observation = env.reset()

state, reward, done, info = env.step((0, 0))
env.render()
