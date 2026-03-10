import gymnasium
import sorting_gym
from sorting_gym.envs.functional_neural_sort_interface import FunctionalNeuralSortInterfaceEnv
from sorting_gym.agents.scripted import quicksort_agent

env = FunctionalNeuralSortInterfaceEnv()
state, info = env.reset()

state, reward, terminated, truncated, info = env.step((0, 0))
env.render()

for i in range(200):
    action = quicksort_agent(state)
    state, reward, terminated, truncated, info = env.step(action)

    if terminated:
        break

env.render()
