import gymnasium
import sorting_gym
from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv
from sorting_gym.envs.wrappers import BoxActionSpaceWrapper

env: BasicNeuralSortInterfaceEnv = gymnasium.make('BasicNeuralSortInterfaceEnv-v0').unwrapped
wrapped = BoxActionSpaceWrapper(env)

observation, info = wrapped.reset()

action = wrapped.action_space_sample()

state, reward, terminated, truncated, info = wrapped.step(action)
env.render()
