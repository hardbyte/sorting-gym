"""Train PPO agents on sorting-gym environments using Stable-Baselines3.

Requires: pip install stable-baselines3

Usage:
    python examples/train_ppo.py                    # Train on all environments
    python examples/train_ppo.py --env knapsack     # Train on one environment
    python examples/train_ppo.py --timesteps 50000  # Custom timestep budget
"""

import argparse

import gymnasium
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv
from sorting_gym.envs.bin_packing import BinPackingEnv
from sorting_gym.envs.job_shop import JobShopSchedulingEnv
from sorting_gym.envs.knapsack import KnapsackEnv
from sorting_gym.envs.wrappers import MultiDiscreteActionSpaceWrapper


def wrap_env(env: gymnasium.Env) -> gymnasium.Env:
    """Wrap an environment for use with SB3: flatten obs + MultiDiscrete actions."""
    return FlattenObservation(MultiDiscreteActionSpaceWrapper(env))


class EpisodeLogCallback(BaseCallback):
    """Log episode statistics periodically."""

    def __init__(self, log_freq=20, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_count = 0
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Check for episode completion via SB3's info dicts
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_count += 1
                self.episode_rewards.append(info["episode"]["r"])
                if self.episode_count % self.log_freq == 0:
                    recent = self.episode_rewards[-self.log_freq :]
                    avg = sum(recent) / len(recent)
                    print(
                        f"  Episode {self.episode_count}: "
                        f"avg reward (last {self.log_freq}) = {avg:.1f}"
                    )
        return True


ENV_CONFIGS = {
    "sort": {
        "make": lambda: BasicNeuralSortInterfaceEnv(k=3),
        "desc": "Sorting (k=3 pointers, small arrays)",
    },
    "knapsack": {
        "make": lambda: KnapsackEnv(k=4, base=20, starting_min_items=4, capacity_ratio=0.5),
        "desc": "0/1 Knapsack (k=4 pointers, small instances)",
    },
    "binpacking": {
        "make": lambda: BinPackingEnv(k=4, base=20, starting_min_items=4),
        "desc": "Bin Packing (k=4 pointers, small instances)",
    },
    "jobshop": {
        "make": lambda: JobShopSchedulingEnv(k=4, base=10, starting_min_jobs=2, num_machines=2),
        "desc": "Job Shop Scheduling (k=4, 2 jobs x 2 machines)",
    },
}


def train(env_name: str, timesteps: int):
    config = ENV_CONFIGS[env_name]
    print(f"\n{'='*60}")
    print(f"Training PPO on: {config['desc']}")
    print(f"{'='*60}")

    env = wrap_env(config["make"]())
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        verbose=0,
    )

    print(f"Training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, callback=EpisodeLogCallback(log_freq=50))

    # Evaluate
    print("\nEvaluating trained agent (10 episodes)...")
    eval_env = wrap_env(config["make"]())
    total_rewards = []
    for ep in range(10):
        obs, _ = eval_env.reset()
        ep_reward = 0
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        total_rewards.append(ep_reward)

    avg = sum(total_rewards) / len(total_rewards)
    print(f"Average reward over 10 episodes: {avg:.1f}")
    print(f"Individual rewards: {[round(r, 1) for r in total_rewards]}")
    eval_env.close()
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train PPO on sorting-gym environments")
    parser.add_argument(
        "--env",
        choices=list(ENV_CONFIGS.keys()) + ["all"],
        default="all",
        help="Which environment to train on (default: all)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20000,
        help="Total training timesteps per environment (default: 20000)",
    )
    args = parser.parse_args()

    envs = list(ENV_CONFIGS.keys()) if args.env == "all" else [args.env]
    for env_name in envs:
        train(env_name, args.timesteps)


if __name__ == "__main__":
    main()
