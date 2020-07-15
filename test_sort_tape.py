
import gym
import sortingenv
from sortingenv.envs.tape import SortTapeAlgorithmicEnv

env: SortTapeAlgorithmicEnv = gym.make('SortTapeAlgorithmicEnv-v0').unwrapped
env.reset()
env.render()

state, reward, done, info = env.step((0,0,0))
print(state)
