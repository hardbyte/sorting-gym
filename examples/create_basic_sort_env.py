import gym
import sorting_gym
from sorting_gym.envs.bubblesort import BubbleInsertionSortInterfaceEnv

env: BubbleInsertionSortInterfaceEnv = gym.make('BasicNeuralSortInterfaceEnv-v0').unwrapped
observation = env.reset()


state, reward, done, info = env.step((0, 0))
env.render()