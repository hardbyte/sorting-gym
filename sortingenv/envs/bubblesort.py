
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
        self.tape_env = SortTapeAlgorithmicEnv(base=base, starting_min_length=4)

        self.INSTRUCTIONS = [
            # Instruction name, arg size in bits
            ('SwapWithNext', k),
            ('MoveVar', k + 1),
            ('AssignVar', 2 * k),

        ]

        self.instruction_space = Discrete(len(self.INSTRUCTIONS))

        self.action_SwapWithNext_space = Discrete(k)
        self.action_MoveVar_space = Tuple([Discrete(k), MultiBinary(1)])
        self.action_AssignVar_space = Tuple([Discrete(k), Discrete(k)])

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
        self.v[1::2] = len(self.tape_env.input_data)

        return self._get_obs()

    def step(self, action):
        pass

    def render(self, mode='human'):
        return self.tape_env.render(mode)
