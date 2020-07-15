from gym.envs.algorithmic import algorithmic_env
from gym.spaces import Discrete, Dict


class SortTapeAlgorithmicEnv(algorithmic_env.TapeAlgorithmicEnv):
    MIN_REWARD_SHORTFALL_FOR_PROMOTION = -.1

    def __init__(self, base=10, starting_min_length=2):
        super().__init__(
            base=base, chars=True, starting_min_length=starting_min_length
        )
        self.last = 500

        self.observation_space = Discrete(self.base + 1)

    def target_from_input_data(self, input_str):
        return list(sorted(input_str))
