import numpy as np
from gym import Env
from gym.spaces import MultiDiscrete

from sorting_gym import DiscreteParametric
from sorting_gym.envs.tape import SortTapeAlgorithmicEnv


class NeuralSortInterfaceEnv(Env):
    """
    Base for Neural interface based environment

    Keeps track of k index variables in `self.v`
    Generates data on environment reset, exposed as `self.A`

    Concrete implementations must implement `step` and create an observation.
    """

    def __init__(self, base, k, instructions):
        self.base = base
        self.k = k
        self.instructions = instructions
        self.v = np.zeros(shape=k, dtype=np.int32)
        self.A = None
        # Generates random data for each episode increasing the length as the agent "levels up"
        self.tape_env = SortTapeAlgorithmicEnv(base=base, starting_min_length=4)

        # Action space is variable - conditioned on the instruction selected
        # This isn't really well supported by the OpenAI Gym api so we've
        # made our own `DiscreteParametric` space class.
        self.action_space = DiscreteParametric(
            len(instructions),
            [instruction.argument_space for instruction in instructions])

    def reset(self):
        self.tape_env.reset()
        self.A = self.tape_env.input_data
        # Reset pointers to low and high
        self.v[::2] = 0
        self.v[1::2] = len(self.A) - 1

    def dispatch(self, instruction, args):
        self.instructions[instruction].implementation(args)
