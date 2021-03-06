import gym
import sorting_gym
from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv
from sorting_gym.envs.wrappers import BoxActionSpaceWrapper

env: BasicNeuralSortInterfaceEnv = gym.make('BasicNeuralSortInterfaceEnv-v0')
wrapped = BoxActionSpaceWrapper(env)

observation = wrapped.reset()

action = wrapped.action_space_sample()

state, reward, done, info = wrapped.step(action)
env.render()
