"""1D Bin Packing environment with pointer-based neural interface.

Items have sizes and must be packed into bins of fixed capacity. The agent
minimizes the number of bins used, receiving constant-size observations
via pointer comparisons.
"""

import numpy as np
from gymnasium.spaces import Discrete, MultiBinary, Tuple

from sorting_gym.envs.basic_neural_sort_interface import Instruction
from sorting_gym.envs.combinatorial_base import NeuralCombinatorialInterfaceEnv


class BinPackingEnv(NeuralCombinatorialInterfaceEnv):
    """1D Bin Packing with constant-size observations via pointer comparisons.

    Instructions:
        0: AssignToExistingBin(i, j) — put item at v[i] into same bin as item at v[j]
        1: AssignToNewBin(i) — open a new bin and assign item at v[i]
        2: MoveVar(i, dir) — move pointer ±1
        3: AssignVar(i, j) — v[i] = v[j]
        4: Finish() — end episode
    """

    def __init__(self, base=100, k=4, starting_min_items=5, bin_capacity=None):
        self._default_bin_capacity = bin_capacity
        self.bin_capacity = 0
        self.assignments = None      # bin index per item, -1 = unassigned
        self.bin_remaining = []      # remaining capacity per bin
        self.num_bins = 0

        instructions = [
            Instruction(0, 'AssignToExistingBin', Tuple([Discrete(k), Discrete(k)]),    self.op_assign_to_existing_bin),
            Instruction(1, 'AssignToNewBin',      Discrete(k),                          self.op_assign_to_new_bin),
            Instruction(2, 'MoveVar',             Tuple([Discrete(k), MultiBinary(1)]), self.op_move_var),
            Instruction(3, 'AssignVar',           Tuple([Discrete(k), Discrete(k)]),    self.op_assign_var),
            Instruction(4, 'Finish',              Discrete(1),                          self.op_finish),
        ]

        # 1 comparison attribute: item size
        super().__init__(
            k=k,
            num_attributes=1,
            instructions=instructions,
            starting_min_items=starting_min_items,
            base=base,
        )

        self._finished = False
        self.reset()

    @property
    def _num_pointer_features(self):
        # Per pointer: is_assigned
        return 1

    @property
    def _num_scalar_features(self):
        # Discretized: num_bins / num_items (4 bins), fraction_unassigned (4 bins)
        return 8

    def _generate_instance(self):
        n = self._get_num_items()
        sizes = self.np_random.integers(1, self.base, size=n).tolist()
        self.items = [(s,) for s in sizes]  # single attribute tuples
        self.num_items = n
        if self._default_bin_capacity is not None:
            self.bin_capacity = self._default_bin_capacity
        else:
            self.bin_capacity = max(1, int(sum(sizes) / 3))
            # Ensure every item can fit in a bin
            self.bin_capacity = max(self.bin_capacity, max(sizes))
        self.assignments = np.full(n, -1, dtype=np.int32)
        self.bin_remaining = []
        self.num_bins = 0
        self._finished = False

    def _get_pointer_features(self):
        feats = np.zeros((self.k, 1), dtype=np.int8)
        for i in range(self.k):
            feats[i, 0] = self.assignments[self.v[i]] >= 0
        return feats

    def _get_scalar_obs(self):
        obs = np.zeros(8, dtype=np.int8)
        # Bins used ratio (4 bins)
        ratio = self.num_bins / max(self.num_items, 1)
        if ratio <= 0.25:
            obs[0] = 1
        elif ratio <= 0.5:
            obs[1] = 1
        elif ratio <= 0.75:
            obs[2] = 1
        else:
            obs[3] = 1
        # Fraction unassigned (4 bins)
        unassigned = int((self.assignments < 0).sum())
        frac = unassigned / max(self.num_items, 1)
        if frac > 0.75:
            obs[4] = 1
        elif frac > 0.5:
            obs[5] = 1
        elif frac > 0.25:
            obs[6] = 1
        else:
            obs[7] = 1
        return obs

    def op_assign_to_existing_bin(self, args):
        i, j = args
        item_idx = self.v[i]
        ref_idx = self.v[j]
        # Item must be unassigned, reference must be assigned
        if self.assignments[item_idx] >= 0 or self.assignments[ref_idx] < 0:
            return  # no-op
        bin_id = self.assignments[ref_idx]
        size = self.items[item_idx][0]
        if size <= self.bin_remaining[bin_id]:
            self.assignments[item_idx] = bin_id
            self.bin_remaining[bin_id] -= size

    def op_assign_to_new_bin(self, args):
        i = args[0] if isinstance(args, (tuple, list)) else args
        item_idx = self.v[i]
        if self.assignments[item_idx] >= 0:
            return  # already assigned
        size = self.items[item_idx][0]
        bin_id = self.num_bins
        self.bin_remaining.append(self.bin_capacity - size)
        self.assignments[item_idx] = bin_id
        self.num_bins += 1

    def op_finish(self, args):
        self._finished = True

    def step(self, action):
        instruction, *args = action
        self.dispatch(instruction, args)

        all_assigned = int((self.assignments < 0).sum()) == 0
        terminated = self._finished or all_assigned
        truncated = False
        reward = -1
        if terminated:
            # Reward is negative of bins used (minimize)
            reward += -self.num_bins

        self.episode_total_reward += reward
        if terminated:
            self.reward_shortfalls.append(self.episode_total_reward)

        info = {
            'num_bins': self.num_bins,
            'num_unassigned': int((self.assignments < 0).sum()),
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        print(f"Items (sizes): {[item[0] for item in self.items]}")
        print(f"Bin capacity: {self.bin_capacity}, Bins used: {self.num_bins}")
        print(f"Assignments: {list(self.assignments)}")
        print(f"Bin remaining: {self.bin_remaining}")
        print(f"Pointers: {list(self.v)}")
