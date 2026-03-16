from gymnasium.spaces import flatdim

from sorting_gym.envs.bin_packing import BinPackingEnv
from sorting_gym.agents.scripted import first_fit_decreasing_agent


def test_bin_packing_creates():
    env = BinPackingEnv(k=4)
    obs, info = env.reset()
    assert obs is not None
    assert 'pointer_comparisons' in obs


def test_bin_packing_observation_constant_size():
    env = BinPackingEnv(k=4, starting_min_items=5)
    size1 = flatdim(env.observation_space)

    env2 = BinPackingEnv(k=4, starting_min_items=20)
    size2 = flatdim(env2.observation_space)

    assert size1 == size2


def test_bin_packing_assign_new_bin():
    env = BinPackingEnv(k=4, starting_min_items=5)
    env.reset()

    obs, reward, terminated, truncated, info = env.step((1, 0))  # AssignToNewBin(0)
    assert env.num_bins == 1
    assert env.assignments[env.v[0]] == 0


def test_bin_packing_assign_existing_bin():
    env = BinPackingEnv(k=4, starting_min_items=5, bin_capacity=1000)
    env.reset()

    # Assign item at pointer 0 to a new bin
    env.step((1, 0))
    assert env.num_bins == 1

    # Move pointer 1 to a different item
    if env.v[1] == env.v[0]:
        env.step((2, 1, True))  # Move pointer 1 right

    # Assign item at pointer 1 to same bin as pointer 0
    env.step((0, 1, 0))  # AssignToExistingBin(1, 0)
    assert env.num_bins == 1  # No new bin opened


def test_bin_packing_finish():
    env = BinPackingEnv(k=4)
    env.reset()
    obs, reward, terminated, truncated, info = env.step((4, 0))
    assert terminated


def test_bin_packing_auto_terminate():
    """Episode ends when all items assigned."""
    env = BinPackingEnv(k=4, starting_min_items=3, bin_capacity=10000)
    env.reset()

    for i in range(env.num_items):
        # Move pointer 0 to item i
        while env.v[0] < i:
            env.step((2, 0, True))
        while env.v[0] > i:
            env.step((2, 0, False))
        obs, reward, terminated, truncated, info = env.step((1, 0))
        if terminated:
            break

    assert terminated


def test_first_fit_decreasing_agent():
    """Test that FFD agent produces valid solutions."""
    env = BinPackingEnv(k=4, base=50, starting_min_items=5)

    for _ in range(50):
        env.reset()
        actions = first_fit_decreasing_agent(env)
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        assert terminated
        assert info['num_unassigned'] == 0, "All items must be assigned"
        # FFD should not use more bins than items
        assert env.num_bins <= env.num_items


def test_bin_packing_action_space_sample():
    env = BinPackingEnv(k=4)
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        assert env.action_space.contains(action)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            env.reset()
