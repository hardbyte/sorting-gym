"""Tape-based sorting environment.

Replaces the old gym.envs.algorithmic.TapeAlgorithmicEnv which was
removed in Gymnasium.  This standalone implementation generates random
integer sequences of increasing length as the agent "levels up".
"""

import gymnasium as gym
from gymnasium.spaces import Discrete


class SortTapeAlgorithmicEnv(gym.Env):
    """Generate random integer sequences for sorting, with progressive difficulty.

    The environment tracks the agent's cumulative reward and promotes to
    longer sequences once performance is good enough.
    """

    MIN_REWARD_SHORTFALL_FOR_PROMOTION = -1

    def __init__(self, base=10, starting_min_length=2):
        super().__init__()
        self.base = base
        self.min_length = starting_min_length
        # Reward bookkeeping for promotion
        self.episode_total_reward = 0
        self.reward_shortfalls = []
        self.last = 500  # episodes at current size before considering promotion

        self.observation_space = Discrete(self.base + 1)
        self.action_space = Discrete(1)  # placeholder; not used directly

        self.input_data = None
        self.target = None

    def _check_levelup(self):
        """Promote to longer sequences when performance is good enough."""
        if len(self.reward_shortfalls) >= self.last:
            recent = self.reward_shortfalls[-self.last:]
            worst = min(recent)
            if worst >= self.MIN_REWARD_SHORTFALL_FOR_PROMOTION:
                self.min_length += 1
                self.reward_shortfalls = []

    def _get_sequence_length(self):
        """Return a random length in [min_length, min_length + offset)."""
        return self.np_random.integers(
            self.min_length,
            self.min_length + max(1, self.min_length // 2) + 1,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._check_levelup()

        length = self._get_sequence_length()
        self.input_data = list(self.np_random.integers(0, self.base, size=length))
        self.target = list(sorted(self.input_data))
        self.episode_total_reward = 0

        obs = 0  # dummy — callers use input_data directly
        return obs, {}

    def step(self, action):
        obs = 0
        reward = 0.0
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def target_from_input_data(self, input_str):
        return list(sorted(input_str))
