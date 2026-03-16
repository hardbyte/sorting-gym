from gymnasium.spaces import flatdim

from sorting_gym.envs.knapsack import KnapsackEnv
from sorting_gym.agents.scripted import greedy_knapsack_agent


def test_knapsack_creates():
    env = KnapsackEnv(k=4)
    obs, info = env.reset()
    assert obs is not None
    assert 'pointer_comparisons' in obs
    assert 'data_comparisons' in obs


def test_knapsack_observation_constant_size():
    """Observation size must not change with different numbers of items."""
    env = KnapsackEnv(k=4, starting_min_items=5)
    obs1, _ = env.reset()
    size1 = flatdim(env.observation_space)

    env2 = KnapsackEnv(k=4, starting_min_items=20)
    obs2, _ = env2.reset()
    size2 = flatdim(env2.observation_space)

    assert size1 == size2, "Observation size must be constant regardless of num_items"


def test_knapsack_select_item():
    env = KnapsackEnv(k=4, starting_min_items=5)
    env.reset()

    # Select item at pointer 0
    obs, reward, terminated, truncated, info = env.step((0, 0))
    assert info['num_selected'] >= 0
    assert reward == -1
    assert not terminated


def test_knapsack_finish():
    env = KnapsackEnv(k=4, starting_min_items=5)
    env.reset()

    # Finish immediately
    obs, reward, terminated, truncated, info = env.step((4, 0))
    assert terminated


def test_knapsack_capacity_respected():
    env = KnapsackEnv(k=4, base=10, starting_min_items=5, capacity_ratio=0.3)
    env.reset()

    _ = env.remaining_capacity
    # Try to select every item by moving pointer and selecting
    for _ in range(env.num_items * 2):
        env.step((0, 0))  # Select at pointer 0
        env.step((2, 0, True))  # Move pointer 0 right

    assert env.remaining_capacity >= 0, "Capacity must never go negative"


def test_knapsack_deselect():
    env = KnapsackEnv(k=4, starting_min_items=5)
    env.reset()

    # Select then deselect
    env.step((0, 0))
    val_after_select = env.total_value
    env.step((1, 0))
    assert env.total_value < val_after_select or val_after_select == 0


def test_greedy_knapsack_agent():
    """Test that the greedy agent produces valid solutions."""
    env = KnapsackEnv(k=4, base=50, starting_min_items=5)

    for _ in range(50):
        env.reset()
        actions = greedy_knapsack_agent(env)
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        assert terminated, "Agent must finish the episode"
        assert env.remaining_capacity >= 0, "Solution must be feasible"
        assert info['total_value'] >= 0


def test_knapsack_action_space_sample():
    env = KnapsackEnv(k=4)
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        assert env.action_space.contains(action)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            env.reset()
