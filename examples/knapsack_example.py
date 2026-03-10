import gymnasium
import sorting_gym
from sorting_gym.envs.knapsack import KnapsackEnv
from sorting_gym.agents.scripted import greedy_knapsack_agent

env = KnapsackEnv(k=4, base=50, starting_min_items=8)
obs, info = env.reset()

print("=== 0/1 Knapsack Environment ===")
env.render()

# Run the greedy agent
actions = greedy_knapsack_agent(env)
for action in actions:
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break

print("\n=== After greedy agent ===")
env.render()
print(f"Total value: {info['total_value']}")
