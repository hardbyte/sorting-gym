import pytest
from gym.spaces import flatten

from sorting_gym.agents.scripted import bubble_sort_agent, insertion_sort_agent, quicksort_agent
from sorting_gym.envs.functional_neural_sort_interface import FunctionalNeuralSortInterfaceEnv
from tests.util import _test_sort_agent


def test_reset_gives_valid_observation():
    env = FunctionalNeuralSortInterfaceEnv(k=4, number_of_functions=5)
    obs = flatten(env.nested_observation_space, env.reset())
    assert obs.shape[0] == 68 + 5 + 6 + 51 + 1


def test_function_env_preserves_function_id():
    """
    Create a functional environment with 2 functions taking 0 args and returning 0 args
    """
    env = FunctionalNeuralSortInterfaceEnv(k=3, number_of_functions=2, function_inputs=0, function_returns=0)
    original_obs = env.reset()
    assert original_obs['current_function'] == -1
    obs, reward, done, info = env.step((3, 0))
    assert obs['current_function'] == 0
    obs, reward, done, info = env.step((3, 1))
    assert obs['current_function'] == 1
    # return
    obs, reward, done, info = env.step((4,))
    assert obs['current_function'] == 0
    obs, reward, done, info = env.step((4,))
    assert obs['current_function'] == -1


def test_function_env_can_pass_through_arg():
    """
    Functional environment with 1 function taking 1 arg and returning 1 arg
    We will create a function that assigns the input to a local variable, and
    returns that local variable.
    """
    env = FunctionalNeuralSortInterfaceEnv(k=3, number_of_functions=1, function_inputs=1, function_returns=1)
    env.reset()
    n = len(env.A) - 1
    assert env.current_function == -1
    assert env.v[1] == n
    assert env.v[2] == 0
    # Call the function 0 with:
    # local variable ID l=0
    # outer variable ID o=1 (pointing to end of array)
    # returning ID r=2
    obs, reward, done, info = env.step((3, 0, 0, 1, 2))
    assert obs['current_function'] == 0
    assert env.v[1] == 0
    assert env.v[2] == 0
    # Now inside the function assign "local" variable (id 1) with the function input (id 0)
    # Which should be our locally passed in end of array pointer
    obs, reward, done, info = env.step((2, 1, 0))
    assert env.v[1] == n
    assert env.v[2] == 0
    # Now return from the function with local variable (id 1).
    # Returning ID is 2, so now v[2] should be n
    obs, reward, done, info = env.step((4, 1))
    assert env.v[2] == n


def test_function_env_swap_args():
    """
    Functional environment with 1 function taking 2 arg and returning 2 args
    We will create a function that swaps the inputs.
    """
    env = FunctionalNeuralSortInterfaceEnv(k=3, number_of_functions=1, function_inputs=2, function_returns=2)
    env.reset()
    n = len(env.A) - 1
    assert env.current_function == -1
    env.v[1] = 1
    env.v[2] = 2
    # Call the function
    obs, reward, done, info = env.step((3, 0,
                                        0, 1, # local inputs
                                        1, 2, # outer variables
                                        1, 2  # write over inputs
                                        ))
    assert obs['current_function'] == 0
    assert env.v[0] == 1
    assert env.v[1] == 2

    # Swap the "local" variables
    # Save temp var (id 2) with the first function input (id 0)
    obs, reward, done, info = env.step((2, 2, 0))
    # Assign v0 = v1
    obs, reward, done, info = env.step((2, 0, 1))
    # Assign v1 = v2
    obs, reward, done, info = env.step((2, 1, 2))

    assert env.v[0] == 2
    assert env.v[1] == 1

    # Now return from the function.
    obs, reward, done, info = env.step((4, 0, 1))

    # Check that the outer scope has had the variables swapped
    assert env.v[1] == 2
    assert env.v[2] == 1


def test_function_env_swap_args_in_call():
    """
    Functional environment with 1 function taking 2 arg and returning 2 args
    We will create a nop function that swaps the inputs by swapping the return args.
    """
    env = FunctionalNeuralSortInterfaceEnv(k=3, number_of_functions=1, function_inputs=2, function_returns=2)
    env.reset()
    n = len(env.A) - 1
    assert env.current_function == -1
    env.v[1] = 1
    env.v[2] = 2
    # Call the function
    obs, reward, done, info = env.step((3, 0,
                                        0, 1, # local inputs
                                        1, 2, # outer variables
                                        1, 2  # write over inputs
                                        ))
    assert obs['current_function'] == 0
    assert env.v[0] == 1
    assert env.v[1] == 2

    # Now return from the function - swapping the return values around
    obs, reward, done, info = env.step((4, 1, 0))

    # Check that the outer scope has had the variables swapped
    assert env.v[1] == 2
    assert env.v[2] == 1


def test_bubble_sort_agent():
    """
    Functional environment should still work using the scripted
    Bubble Sort agent.
    """
    env = FunctionalNeuralSortInterfaceEnv(k=3)
    agent_f = bubble_sort_agent
    _test_sort_agent(agent_f, env, 100)


def test_bubble_sort_agent_not_enough_pointers():
    env = FunctionalNeuralSortInterfaceEnv(k=2)
    agent_f = bubble_sort_agent
    with pytest.raises(IndexError):
        _test_sort_agent(agent_f, env, 100)


def test_quick_sort_agent():
    """
    Tests the environment using a Quick Sort agent.

    c.f. Algorithm 8 - pg 25
    """
    env = FunctionalNeuralSortInterfaceEnv(k=4, number_of_functions=2)
    _test_sort_agent(quicksort_agent, env, number_of_problems=100, max_steps=10000, verbose=True)
