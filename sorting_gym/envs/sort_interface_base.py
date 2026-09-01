import numpy as np
from gymnasium import Env

from sorting_gym import DiscreteParametric
from sorting_gym.envs.tape import SortTapeAlgorithmicEnv


class NeuralSortInterfaceEnv(Env):
    """
    Base for Neural interface based environment

    Keeps track of k index variables in `self.v`
    Generates data on environment reset, exposed as `self.A`

    Concrete implementations must implement `step` and create an observation.
    """

    def __init__(self, base, k, instructions, max_episode_steps=None):
        self.base = base
        self.k = k
        self.instructions = instructions
        self.v = np.zeros(shape=k, dtype=np.int32)
        self.A = None
        # None means "scale the budget with the instance": an O(n^2) sort needs
        # roughly 0.5*n^2 instructions, so 4*n^2 leaves plenty of headroom.
        self.max_episode_steps = max_episode_steps
        self.steps_taken = 0
        # Generates random data for each episode increasing the length as the agent "levels up"
        self.tape_env = SortTapeAlgorithmicEnv(base=base, starting_min_length=4)

        # Action space is variable - conditioned on the instruction selected
        # This isn't really well supported by the Gymnasium api so we've
        # made our own `DiscreteParametric` space class.
        self.action_space = DiscreteParametric(
            len(instructions),
            [instruction.argument_space for instruction in instructions])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.tape_env.reset(seed=seed)
        self.A = self.tape_env.input_data
        # Reset pointers to low and high
        self.v[::2] = 0
        self.v[1::2] = len(self.A) - 1
        self.steps_taken = 0

    @property
    def step_budget(self):
        if self.max_episode_steps is not None:
            return self.max_episode_steps
        return 4 * len(self.A) ** 2

    def _account_for_step(self, terminated):
        """Count the step, decide truncation, and feed the difficulty curriculum.

        Returns `truncated`. Must be called exactly once per `step`.
        """
        self.steps_taken += 1
        self.tape_env.episode_total_reward -= 1
        truncated = not terminated and self.steps_taken >= self.step_budget
        if terminated or truncated:
            self.tape_env.record_episode_outcome(solved=terminated)
        return truncated

    def dispatch(self, instruction, args):
        self.instructions[instruction].implementation(args)
