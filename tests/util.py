import pytest


def _test_sort_agent(agent_f, env, number_of_problems=1000, max_steps = 1000, verbose=False):
    k = env.k
    for problem in range(1, number_of_problems):
        obs = env.reset()
        for step in range(1, max_steps+1):
            action = agent_f(obs, k)
            obs, reward, is_done, info = env.step(action)
            if is_done:
                if verbose:
                    print(f"Solved problem {problem} of size {len(env.A)} in {step} steps")
                break
            if step == max_steps:
                pytest.fail(f"Didn't solve problem {problem} of size {len(env.A)}.")
