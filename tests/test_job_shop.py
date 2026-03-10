from gymnasium.spaces import flatdim

from sorting_gym.envs.job_shop import JobShopSchedulingEnv
from sorting_gym.agents.scripted import spt_scheduling_agent


def test_job_shop_creates():
    env = JobShopSchedulingEnv(k=4, num_machines=2)
    obs, info = env.reset()
    assert obs is not None
    assert 'pointer_comparisons' in obs


def test_job_shop_observation_constant_size():
    env = JobShopSchedulingEnv(k=4, starting_min_jobs=2, num_machines=2)
    size1 = flatdim(env.observation_space)

    env2 = JobShopSchedulingEnv(k=4, starting_min_jobs=10, num_machines=2)
    size2 = flatdim(env2.observation_space)

    assert size1 == size2


def test_job_shop_schedule_operation():
    env = JobShopSchedulingEnv(k=4, num_machines=2)
    env.reset()

    obs, reward, terminated, truncated, info = env.step((0, 0))  # ScheduleNext(0)
    assert info['num_scheduled'] == 1
    assert env.makespan > 0


def test_job_shop_finish():
    env = JobShopSchedulingEnv(k=4, num_machines=2)
    env.reset()
    obs, reward, terminated, truncated, info = env.step((3, 0))  # Finish
    assert terminated


def test_job_shop_auto_terminate():
    """Episode ends when all operations scheduled."""
    env = JobShopSchedulingEnv(k=4, starting_min_jobs=2, num_machines=2)
    env.reset()

    for i in range(env.num_items):
        while env.v[0] < i:
            env.step((1, 0, True))
        while env.v[0] > i:
            env.step((1, 0, False))
        obs, reward, terminated, truncated, info = env.step((0, 0))
        if terminated:
            break

    assert terminated
    assert info['num_scheduled'] == env.num_items


def test_spt_scheduling_agent():
    """Test that SPT agent produces valid schedules."""
    env = JobShopSchedulingEnv(k=4, base=20, starting_min_jobs=2, num_machines=2)

    for _ in range(50):
        env.reset()
        actions = spt_scheduling_agent(env)
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        assert terminated
        assert info['num_scheduled'] == env.num_items, "All operations must be scheduled"
        assert env.makespan > 0


def test_job_shop_action_space_sample():
    env = JobShopSchedulingEnv(k=4, num_machines=2)
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        assert env.action_space.contains(action)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            env.reset()


def test_job_shop_different_machine_counts():
    """Test with varying machine counts."""
    for num_machines in [2, 3, 4]:
        env = JobShopSchedulingEnv(k=4, num_machines=num_machines, starting_min_jobs=3)
        obs, info = env.reset()
        assert env.num_items == env.num_jobs * num_machines
