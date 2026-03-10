import gymnasium
import sorting_gym
from sorting_gym.envs.bin_packing import BinPackingEnv
from sorting_gym.agents.scripted import first_fit_decreasing_agent

env = BinPackingEnv(k=4, base=50, starting_min_items=8)
obs, info = env.reset()

print("=== 1D Bin Packing Environment ===")
env.render()

# Run the FFD agent
actions = first_fit_decreasing_agent(env)
for action in actions:
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break

print("\n=== After FFD agent ===")
env.render()
print(f"Bins used: {info['num_bins']}")
