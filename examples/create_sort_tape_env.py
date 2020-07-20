import gym
import sorting_gym
from sorting_gym.envs.tape import SortTapeAlgorithmicEnv

env: SortTapeAlgorithmicEnv = gym.make('SortTapeAlgorithmicEnv-v0').unwrapped
observation = env.reset()
env.render()

state, reward, done, info = env.step((0, 0, 0))
