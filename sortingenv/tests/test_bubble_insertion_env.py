import pytest
import numpy as np
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
    k = 4
    env = BubbleInsertionSortInterfaceEnv(k=k)



    def _ugly_i_stride(i, k):
        i_stride = 0
        for tmpi in range(i, -1, -1):
            i_stride += k - (i + 1)
        return i_stride * 6

    def v_less_than(observation, i, j) -> bool:
        cmps = observation['pairwise_view_comparisons']
        i_stride = _ugly_i_stride(i, k)
        return cmps[i_stride * i + (j-1) * 6 + 0]

    def v_equals(observation, i, j) -> bool:
        cmps = observation['pairwise_view_comparisons']
        i_stride = _ugly_i_stride(i, k)
        return cmps[i_stride * i + (j - 1) * 6 + 1]

    def data_neighbour_greater(obs, i, direction):
        cmps = obs['neighbour_view_comparisons']
        offset = 1 if direction == -1 else 4
        return cmps[8*i + offset]

    # Return instructions
    def SwapWithNext(i):
        return 0, i

    def MoveVar(i, direction):
        return 1, i, direction > 0.5

    def AssignVar(a, b):
        return 2, a, b

    def bubble_sort_agent(obs):
        i, j, l = 0, 1, 2
        if v_less_than(obs, i, j):
            if data_neighbour_greater(obs, i, +1):
                return SwapWithNext(i)
            else:
                return MoveVar(i, +1)
        elif v_equals(obs, i, j):
            return MoveVar(j, -1)
        else:
            return AssignVar(i, l)

    for problem in range(1, 2000):
        obs = env.reset()

        for step in range(1000):
            action = bubble_sort_agent(obs)
            obs, reward, is_done, info = env.step(action)
            #print(info)
            if is_done:
                print(f"Solved problem {problem} of size {len(env.A)} in {step} steps")
                break
        if step == 999:
            print("Didn't solve?", env.A, env.v)


if __name__ == '__main__':
    test_bubble_sort_agent()