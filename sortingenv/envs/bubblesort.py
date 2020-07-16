
from gym import Env
from gym.spaces import Discrete, Dict, MultiBinary, flatten_space, Tuple, MultiDiscrete
import numpy as np

from sortingenv.envs.tape import SortTapeAlgorithmicEnv


class BubbleInsertionSortInterfaceEnv(Env):
    """
    Neural Computer based environment
    c.f. Section 4.1.2

    Keeps track of k index variables $v_1, .., v_k$

    Actions:
    - SwapWithNext(i) which swaps A[v_i] and A[v_i + 1];
    - MoveVar(i, +/- 1) which increments or decrements v_i
      bounded by start and end of the view.
    - AssignVar(i, j) which assigns v_i = v_j

    """

    def __init__(self, base=10, k=4):
        self.k = k
        self.v = np.zeros(shape=k, dtype=np.int32)
        self.base = base
        # Generates random data for each episode increasing the length as the agent "levels up"
        self.tape_env = SortTapeAlgorithmicEnv(base=base, starting_min_length=4)

        self.INSTRUCTIONS = [
            # Instruction name, arg size in bits
            ('SwapWithNext', k),
            ('MoveVar', k + 1),
            ('AssignVar', 2 * k),
        ]

        # Action space is variable, which isn't really supported by OpenAI Gym
        self.instruction_space = Discrete(len(self.INSTRUCTIONS))

        self.action_SwapWithNext_space = Discrete(k)
        self.action_MoveVar_space = Tuple([Discrete(k), MultiBinary(1)])
        self.action_AssignVar_space = Tuple([Discrete(k), Discrete(k)])

        self.action_space = Tuple([self.instruction_space])

        """
        Observation:

        - Comparisons between data view pairs (<, ==, >) for both index and data
        - Comparisons to neighbours in original data.

        """
        self.nested_observation_space = Dict(
            pairwise_view_comparisons=MultiBinary((6 * k) * (k-1)//2),
            neighbour_view_comparisons=MultiBinary((4 * k) * 2),
        )
        # self.observation_space = flatten_space(self.nested_observation_space)
        self.observation_space = self.nested_observation_space

        self.reset()

    def _get_obs(self):
        A = self.tape_env.input_data
        # TODO return read comparisons etc
        return self.observation_space.sample()

    def reset(self):
        self.tape_env.reset()
        # Reset pointers to low and high
        self.v[::2] = 0
        self.v[1::2] = len(self.tape_env.input_data) - 1

        return self._get_obs()

    def step(self, action):
        instruction, *args = action
        A = self.tape_env.input_data
        if instruction == 0:
            # SwapWithNext(i)
            # swaps A[v_i] and A[v_i + 1]
            i = args[0]
            v_i = self.v[i]
            v_i_next = min(len(A) - 1, v_i + 1)
            assert v_i < len(A), f"Expected v_i ({v_i}) to be less than len(A) ({len(A)})"
            assert v_i_next < len(A)
            A[v_i], A[v_i_next] = A[v_i_next], A[v_i]

        elif instruction == 1:
            # MoveVar(i, +/- 1)
            # increments or decrements v_i
            # bounded by start and end of the view.
            i, direction = args
            if direction:
                self.v[i] = min(self.v[i] + 1, len(A) - 1)
            else:
                self.v[i] = max(self.v[i] - 1, 0)

        elif instruction == 2:
            # AssignVar(i, j)
            # assigns v_i = v_j
            i, j = args
            self.v[i] = self.v[j]

        else:
            raise ValueError()
        return self._get_obs()

    def render(self, mode='human'):
        return self.tape_env.render(mode)
