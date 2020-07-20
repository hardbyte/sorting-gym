import pytest
import numpy as np
from gym.spaces import flatten, flatdim

from sorting_gym.envs.bubblesort import BubbleInsertionSortInterfaceEnv


def test_observation_size():
    env = BubbleInsertionSortInterfaceEnv(k=4)
    assert flatdim(env.observation_space) == 68


def test_reset_gives_valid_observation():
    env = BubbleInsertionSortInterfaceEnv(k=4)
    obs = flatten(env.nested_observation_space, env.reset())
    assert obs.shape[0] == 68


def _get_pairwise_offset(i, j, k):
    i_jmp = 0
    for tmpi in range(i-1, -1, -1):
        i_jmp += k - (tmpi + 1)
    return i_jmp * 6 + (j - i -1) * 6


def v_less_than(observation, i, j, k) -> bool:
    cmps = observation['pairwise_view_comparisons']
    offset = _get_pairwise_offset(i, j, k)
    return cmps[offset + 0]


def v_greater_than(observation, i, j, k) -> bool:
    cmps = observation['pairwise_view_comparisons']
    offset = _get_pairwise_offset(i, j, k)
    return cmps[offset + 2]


def v_equals(observation, i, j, k) -> bool:
    cmps = observation['pairwise_view_comparisons']
    offset = _get_pairwise_offset(i, j, k)
    return cmps[offset + 1]


def data_neighbour_greater(obs, i, direction):
    cmps = obs['neighbour_view_comparisons']
    offset = 1 if direction == -1 else 4
    return cmps[8*i + offset]


def data_neighbour_less(obs, i, direction):
    cmps = obs['neighbour_view_comparisons']
    offset = 3 if direction == -1 else 6
    return cmps[8*i + offset]


# Return instructions
###########################

def SwapWithNext(i):
    return 0, i


def MoveVar(i, direction):
    return 1, i, direction > 0.5


def AssignVar(a, b):
    return 2, a, b


def test_bubble_sort_agent():
    """
    Tests the environment using a Bubble Sort agent.

    c.f. Algorithm 2 - pg 19
    """
    k = 4
    env = BubbleInsertionSortInterfaceEnv(k=k)

    def bubble_sort_agent(obs):
        i, j, l = 0, 1, 2
        if v_less_than(obs, i, j, k):
            if data_neighbour_greater(obs, i, +1):
                return SwapWithNext(i)
            else:
                return MoveVar(i, +1)
        elif v_equals(obs, i, j, k):
            return MoveVar(j, -1)
        else:
            return AssignVar(i, l)

    agent_f = bubble_sort_agent

    _test_sort_agent(agent_f, env, 2000)


def _test_sort_agent(agent_f, env, number_of_problems=1000, max_steps = 1000, verbose=False):
    for problem in range(1, number_of_problems):
        obs = env.reset()
        for step in range(1, max_steps+1):
            action = agent_f(obs)
            if verbose: print("Action: ", action)
            obs, reward, is_done, info = env.step(action)
            if verbose: print(info)
            if is_done:
                if verbose:
                    print(f"Solved problem {problem} of size {len(env.A)} in {step} steps")
                break
            if step == max_steps:
                pytest.fail(f"Didn't solve problem {problem} of size {len(env.A)}.")


def test_insertion_sort_agent():
    """
    Tests the environment using an Insertion Sort agent.

    c.f. Algorithm 4 - pg 20
    """
    k = 4
    env = BubbleInsertionSortInterfaceEnv(k=k)

    def insertion_sort_agent(obs):
        i, j = 0, 1
        low = 2
        # Set initial value of vj
        if v_less_than(obs, i, j, k):
            return AssignVar(j, i)
        elif v_equals(obs, i, j, k):
            return MoveVar(i, +1)
        else:
            if data_neighbour_greater(obs, j, +1):
                return SwapWithNext(j)
            elif v_greater_than(obs, j, low, k) and data_neighbour_less(obs, j, -1):
                return MoveVar(j, -1)
            else:
                return AssignVar(j, i)

    _test_sort_agent(insertion_sort_agent, env, verbose=False)


if __name__ == '__main__':
    #test_bubble_sort_agent()
    test_insertion_sort_agent()