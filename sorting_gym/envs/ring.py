"""Sorting on a ring: the same interface with the edges removed.

`MoveVar` and `SwapWithNext` wrap around, so no position is special and the
`at_left_edge` / `at_right_edge` bits are always zero. The goal is for A to be
*any rotation* of sorted order, which is the only sensible target once there is
no first element.

The point is to separate recalling an algorithm from deriving one. The array
task and the ring task carry identical information -- the same pointers, the
same comparison bits, the same instruction set -- but every textbook sort, and
the policy an LLM wrote for the array env, anchors on the left edge. On a ring
that anchor does not exist, so a policy has to park a pointer to mark an
arbitrary seam and never swap across it. Measured on the reference policy that
is still O(n^2), so the task is no harder in complexity, only unfamiliar.
"""

from collections import OrderedDict

import numpy as np

from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv


def descents(array):
    """Adjacent drops counted around the ring."""
    n = len(array)
    return sum(1 for i in range(n) if array[i] > array[(i + 1) % n])


def is_rotation_of_sorted(array):
    """True when the ring reads non-decreasing from some starting point.

    A sorted ring has exactly one drop, where the largest element meets the
    smallest; an all-equal ring has none.
    """
    return descents(array) <= 1


class RingSortInterfaceEnv(BasicNeuralSortInterfaceEnv):
    """Ring variant of `BasicNeuralSortInterfaceEnv`."""

    def _get_obs(self):
        """Parent observation with the neighbour bits recomputed to wrap."""
        observation = super()._get_obs()
        n = len(self.A)
        neighbours = np.zeros((self.k, 8), dtype=np.int8)
        for i in range(self.k):
            position = int(self.v[i])
            left = self.A[(position - 1) % n]
            right = self.A[(position + 1) % n]
            current = self.A[position]
            # Bits 0 and 7 stay clear: on a ring there is no first or last.
            neighbours[i, 1] = current > left
            neighbours[i, 2] = current == left
            neighbours[i, 3] = current < left
            neighbours[i, 4] = current > right
            neighbours[i, 5] = current == right
            neighbours[i, 6] = current < right
        return OrderedDict([
            ('neighbour_view_comparisons', neighbours.flatten()),
            ('pairwise_view_comparisons', observation['pairwise_view_comparisons']),
        ])

    def op_move_var(self, args):
        i, direction = args
        n = len(self.A)
        self.v[i] = (self.v[i] + (1 if direction else -1)) % n

    def op_swap_with_next(self, args):
        i = args[0]
        n = len(self.A)
        position = int(self.v[i])
        following = (position + 1) % n
        self.A[position], self.A[following] = self.A[following], self.A[position]

    def step(self, action):
        instruction, *args = action
        self.dispatch(instruction, args)

        terminated = is_rotation_of_sorted(self.A)
        cost = self.instruction_cost(instruction)
        self.episode_cost += cost
        truncated = self._account_for_step(terminated)
        info_dict = {'data': list(self.A), 'interface': list(self.v),
                     'cost': cost, 'episode_cost': self.episode_cost}
        return self._get_obs(), -cost, terminated, truncated, info_dict
