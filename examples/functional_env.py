
import gym
import sorting_gym
from sorting_gym.envs.functional_neural_sort_interface import FunctionalNeuralSortInterfaceEnv
from sorting_gym.agents.scripted import quicksort_agent

env = FunctionalNeuralSortInterfaceEnv()
observation = env.reset()

state, reward, done, info = env.step((0, 0))
env.render()

for i in range(200):
    action = quicksort_agent(state)
    state, reward, is_done, info = env.step(action)
    # print(info)

    if is_done:
        break

env.render()