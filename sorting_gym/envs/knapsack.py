"""0/1 Knapsack environment with pointer-based neural interface.

Items have weight and value. The agent uses pointer variables to scan items
and decide which to select, receiving only constant-size pairwise comparison
observations regardless of the number of items.
"""

import numpy as np
from gymnasium.spaces import Discrete, MultiBinary, Tuple

from sorting_gym.envs.basic_neural_sort_interface import Instruction
from sorting_gym.envs.combinatorial_base import NeuralCombinatorialInterfaceEnv


class KnapsackEnv(NeuralCombinatorialInterfaceEnv):
    """0/1 Knapsack with constant-size observations via pointer comparisons.

    Items have weight, value, and derived value/weight ratio (3 comparison attributes).
    The agent selects items without exceeding capacity, then calls Finish.

    Instructions:
        0: SelectItem(i) — select item at pointer v[i]
        1: DeselectItem(i) — deselect item at pointer v[i]
        2: MoveVar(i, dir) — move pointer ±1
        3: AssignVar(i, j) — v[i] = v[j]
        4: Finish() — end episode
    """

    def __init__(self, base=100, k=4, starting_min_items=5, capacity_ratio=0.5):
        self.capacity_ratio = capacity_ratio
        self.capacity = 0
        self.remaining_capacity = 0
        self.selected = None
        self.total_value = 0

        instructions = [
            Instruction(0, 'SelectItem',   Discrete(k),                          self.op_select_item),
            Instruction(1, 'DeselectItem', Discrete(k),                          self.op_deselect_item),
            Instruction(2, 'MoveVar',      Tuple([Discrete(k), MultiBinary(1)]), self.op_move_var),
            Instruction(3, 'AssignVar',    Tuple([Discrete(k), Discrete(k)]),    self.op_assign_var),
            Instruction(4, 'Finish',       Discrete(1),                          self.op_finish),
        ]

        # 3 comparison attributes: weight, value, value/weight ratio
        super().__init__(
            k=k,
            num_attributes=3,
            instructions=instructions,
            starting_min_items=starting_min_items,
            base=base,
        )

        self._finished = False
        self.reset()

    @property
    def _num_pointer_features(self):
        # Per pointer: is_selected, does_it_fit
        return 2

    @property
    def _num_scalar_features(self):
        # Discretized remaining_capacity / total_capacity in 4 bins
        return 4

    def _generate_instance(self):
        n = self._get_num_items()
        weights = self.np_random.integers(1, self.base, size=n)
        values = self.np_random.integers(1, self.base, size=n)
        self.items = list(zip(weights.tolist(), values.tolist()))
        self.num_items = n
        self.capacity = max(1, int(sum(weights) * self.capacity_ratio))
        self.remaining_capacity = self.capacity
        self.selected = np.zeros(n, dtype=bool)
        self.total_value = 0
        self._finished = False

    def _get_item_attribute(self, item_idx, attr_idx):
        w, v = self.items[item_idx]
        if attr_idx == 0:
            return w
        elif attr_idx == 1:
            return v
        else:
            # value/weight ratio — scale by 100 for integer comparison
            return int(v * 100 / max(w, 1))

    def _get_pointer_features(self):
        feats = np.zeros((self.k, 2), dtype=np.int8)
        for i in range(self.k):
            idx = self.v[i]
            feats[i, 0] = self.selected[idx]
            feats[i, 1] = self.items[idx][0] <= self.remaining_capacity
        return feats

    def _get_scalar_obs(self):
        # Discretize remaining_capacity / capacity into 4 bins
        ratio = self.remaining_capacity / max(self.capacity, 1)
        bins = np.zeros(4, dtype=np.int8)
        if ratio > 0.75:
            bins[0] = 1
        elif ratio > 0.5:
            bins[1] = 1
        elif ratio > 0.25:
            bins[2] = 1
        else:
            bins[3] = 1
        return bins

    def op_select_item(self, args):
        i = args[0] if isinstance(args, (tuple, list)) else args
        idx = self.v[i]
        w, v = self.items[idx]
        if not self.selected[idx] and w <= self.remaining_capacity:
            self.selected[idx] = True
            self.remaining_capacity -= w
            self.total_value += v

    def op_deselect_item(self, args):
        i = args[0] if isinstance(args, (tuple, list)) else args
        idx = self.v[i]
        if self.selected[idx]:
            w, v = self.items[idx]
            self.selected[idx] = False
            self.remaining_capacity += w
            self.total_value -= v

    def op_finish(self, args):
        self._finished = True

    def step(self, action):
        instruction, *args = action
        self.dispatch(instruction, args)

        terminated = self._finished
        truncated = False
        reward = -1
        if terminated:
            reward += self.total_value

        self.episode_total_reward += reward
        if terminated:
            self.reward_shortfalls.append(self.episode_total_reward)

        info = {
            'total_value': self.total_value,
            'remaining_capacity': self.remaining_capacity,
            'num_selected': int(self.selected.sum()),
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode="human"):
        """Render current state.

        mode='human': print to stdout.
        mode='rgb_array': return a text-art string for frame capture.
        """
        if mode == "human":
            print(self.render(mode="rgb_array"))
            return None

        n = self.num_items
        lines = []

        # Capacity bar
        used = self.capacity - self.remaining_capacity
        cap_pct = used / max(self.capacity, 1)
        bar_len = 30
        filled = int(cap_pct * bar_len)
        cap_bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
        lines.append(f"Capacity: [{cap_bar}] {used}/{self.capacity}")
        lines.append(f"Value collected: {self.total_value}")
        lines.append("")

        # Item table
        lines.append("  #  W   V  V/W  Status   Ptrs")
        lines.append(" " + "\u2500" * 36)
        for i in range(n):
            w, v = self.items[i]
            ratio = v / max(w, 1)
            sel = "\u2713 sel" if self.selected[i] else "     "
            fits = "" if self.selected[i] else ("\u2717" if w > self.remaining_capacity else " ")
            ptrs = ",".join(f"v{p}" for p in range(self.k) if self.v[p] == i)
            marker = "\u25b6" if ptrs else " "
            lines.append(f" {marker}{i:2d} {w:3d} {v:3d} {ratio:4.1f}  {sel}{fits}  {ptrs}")

        lines.append(" " + "\u2500" * 36)
        if self._finished:
            lines.append(" DONE!")
        return "\n".join(lines)
