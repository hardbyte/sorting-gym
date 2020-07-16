import pytest
from gym.spaces import flatten, flatdim

from sortingenv.envs.bubblesort import BubbleInsertionSortInterfaceEnv


def test_observation_size():
    env = BubbleInsertionSortInterfaceEnv(k=4)
    assert flatdim(env.observation_space) == 68


def test_reset_gives_valid_observation():
    env = BubbleInsertionSortInterfaceEnv(k=4)
    obs = flatten(env.nested_observation_space, env.reset())
    assert obs.shape[0] == 68


def test_bubble_sort_agent():
    env = BubbleInsertionSortInterfaceEnv(k=4)

    obs = env.reset()

    def v_less_than(observation, i, j) -> bool:
        cmps = observation['pairwise_view_comparisons']
        stride = 6
        return cmps[i-1 + stride*j]

    def v_equals(observation, i, j) -> bool:
        cmps = observation['pairwise_view_comparisons']
        stride = 6
        return cmps[i + stride*j]

    def data_neigbour_greater(obs, i, direction):
        cmps = obs['neighbour_view_comparisons']
        stride = 0 if direction == -1 else 4
        return cmps[i + stride]

    # Return instructions
    def SwapWithNext(i):
        return 0, i

    def MoveVar(i, direction):
        return 1, i, direction > 0.5

    def AssignVar(a, b):
        return 2, a, b

    def bubble_sort_agent(obs):
        i, j, l = 1, 2, 3
        if v_less_than(obs, i, j):
            if data_neigbour_greater(obs, i, +1):
                return SwapWithNext(i)
            else:
                return MoveVar(i, +1)
        elif v_equals(obs, i, j):
            return MoveVar(j, -1)
        else:
            return AssignVar(i, l)

    for step in range(5):
        action = bubble_sort_agent(obs)
        print(action)
        obs = env.step(action)


if __name__ == '__main__':
    test_bubble_sort_agent()