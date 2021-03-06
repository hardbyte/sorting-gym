from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

from gym import Space
from gym.spaces import Discrete, Dict, MultiBinary, Tuple
import numpy as np

from sorting_gym.envs.sort_interface_base import NeuralSortInterfaceEnv


@dataclass(frozen=True)
class Instruction:
    """
    Keeps track of an instruction's:
    opcode, name, argument space
    """
    opcode: int
    name: str
    argument_space: Space
    implementation: Callable

    def __repr__(self):
        return f"<Instruction {self.opcode} - {self.name}>"


class BasicNeuralSortInterfaceEnv(NeuralSortInterfaceEnv):
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

        instructions = [
            # Instruction(opcode, name, argument space, implementation method)
            Instruction(0, 'SwapWithNext', Discrete(k),                             self.op_swap_with_next),
            Instruction(1, 'MoveVar',      Tuple([Discrete(k), MultiBinary(1)]),    self.op_move_var),
            Instruction(2, 'AssignVar',    Tuple([Discrete(k), Discrete(k)]),       self.op_assign_var),
        ]
        # super call will add the DiscreteParametric action_space attribute
        super().__init__(base, k, instructions)

        self.nested_observation_space = Dict(
            pairwise_view_comparisons=MultiBinary((6 * k) * (k-1)//2),
            neighbour_view_comparisons=MultiBinary((4 * k) * 2),
        )
        # self.observation_space = flatten_space(self.nested_observation_space)
        self.observation_space = self.nested_observation_space

        self.reset()

    def _get_obs(self):
        k = self.k
        neighbour_comparisons = np.zeros((k, 8), dtype=np.int8)
        pairwise_comparisons = np.zeros((6 * k) * (k - 1) // 2, dtype=np.int8)
        for i in range(k):
            neighbour_comparisons[i, :] = [1, 0, 0, 0, 0, 0, 0, 1]
            if self.v[i] > 0:
                neighbour_comparisons[i, 0] = 0
                neighbour_comparisons[i, 1] = self.A[self.v[i]] > self.A[self.v[i] - 1]
                neighbour_comparisons[i, 2] = self.A[self.v[i]] == self.A[self.v[i] - 1]
                neighbour_comparisons[i, 3] = self.A[self.v[i]] < self.A[self.v[i] - 1]
            if self.v[i] + 1 < len(self.A):
                neighbour_comparisons[i, 4] = self.A[self.v[i]] > self.A[self.v[i] + 1]
                neighbour_comparisons[i, 5] = self.A[self.v[i]] == self.A[self.v[i] + 1]
                neighbour_comparisons[i, 6] = self.A[self.v[i]] < self.A[self.v[i] + 1]
                neighbour_comparisons[i, 7] = 0

            # Pairwise comparisons
            if i < k-1:
                # yeah this is not pretty
                i_jmp = 0
                for tmpi in range(i-1, -1, -1):
                    i_jmp += k - (tmpi + 1)
                i_jmp *= 6

                for j in range(i+1, k):
                    j_stride = j - i - 1
                    pairwise_comparisons[i_jmp + (j_stride)*6 + 0] = self.v[i] < self.v[j]
                    pairwise_comparisons[i_jmp + (j_stride)*6 + 1] = self.v[i] == self.v[j]
                    pairwise_comparisons[i_jmp + (j_stride)*6 + 2] = self.v[i] > self.v[j]
                    pairwise_comparisons[i_jmp + (j_stride)*6 + 3] = self.A[self.v[i]] < self.A[self.v[j]]
                    pairwise_comparisons[i_jmp + (j_stride)*6 + 4] = self.A[self.v[i]] == self.A[self.v[j]]
                    pairwise_comparisons[i_jmp + (j_stride)*6 + 5] = self.A[self.v[i]] > self.A[self.v[j]]

        return OrderedDict([('neighbour_view_comparisons', neighbour_comparisons.flatten()),
                            ('pairwise_view_comparisons', pairwise_comparisons)])

    def reset(self):
        super().reset()
        return self._get_obs()

    def op_swap_with_next(self, args):
        # SwapWithNext(i)
        # swaps A[v_i] and A[v_i + 1]
        i = args[0]
        v_i = self.v[i]
        v_i_next = min(len(self.A) - 1, v_i + 1)
        assert v_i < len(self.A), f"Expected v_i ({v_i}) to be less than len(A) ({len(self.A)})"
        assert v_i_next < len(self.A)
        self.A[v_i], self.A[v_i_next] = self.A[v_i_next], self.A[v_i]

    def op_move_var(self, args):
        # MoveVar(i, +/- 1)
        # increments or decrements v_i
        # bounded by start and end of the view.
        i, direction = args
        if direction:
            self.v[i] = min(self.v[i] + 1, len(self.A) - 1)
        else:
            self.v[i] = max(self.v[i] - 1, 0)

    def op_assign_var(self, args):
        # AssignVar(i, j)
        # assigns v_i = v_j
        i, j = args
        self.v[i] = self.v[j]

    def step(self, action):
        instruction, *args = action
        self.dispatch(instruction, args)

        # Check for solved, calculate reward
        done = self.A == self.tape_env.target
        if done:
            # So the strings get longer
            self.tape_env.episode_total_reward = len(self.A)
        reward = -1
        info_dict = {'data': self.A, 'interface': self.v}
        return self._get_obs(), reward, done, info_dict

    def render(self, mode='human'):
        return self.tape_env.render(mode)


