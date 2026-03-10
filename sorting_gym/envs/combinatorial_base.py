"""Base class for neural combinatorial optimization environments.

Extends the pointer-based neural interface design from sorting to general
combinatorial optimization problems. Items have multiple attributes (e.g.
weight, value for knapsack), and observations are constant-size pairwise
comparisons between pointer-referenced items.
"""

from collections import OrderedDict

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Dict, MultiBinary

from sorting_gym import DiscreteParametric
from sorting_gym.envs.basic_neural_sort_interface import Instruction


class NeuralCombinatorialInterfaceEnv(gym.Env):
    """Base class for pointer-based combinatorial optimization environments.

    Subclasses must implement:
        _generate_instance() — populate self.items and any problem-specific state
        _get_pointer_features() — per-pointer binary features (e.g. selected, fits)
        _get_scalar_obs() — global scalar features
        _num_pointer_features — property returning count of per-pointer binary features
        _num_scalar_features — property returning count of scalar features
        step(action) — execute action and return (obs, reward, terminated, truncated, info)
    """

    # Progressive difficulty parameters
    MIN_REWARD_SHORTFALL_FOR_PROMOTION = -1
    PROMOTION_WINDOW = 200

    def __init__(self, k, num_attributes, instructions, starting_min_items=5, base=100):
        super().__init__()
        self.k = k
        self.num_attributes = num_attributes
        self.base = base
        self.instructions = instructions

        # Pointer variables
        self.v = np.zeros(shape=k, dtype=np.int32)

        # Items: 2D array (num_items × num_attributes), set on reset
        self.items = None
        self.num_items = 0

        # Progressive difficulty
        self.min_num_items = starting_min_items
        self.episode_total_reward = 0
        self.reward_shortfalls = []

        # Action space
        self.action_space = DiscreteParametric(
            len(instructions),
            [instruction.argument_space for instruction in instructions])

        # Observation space — constant size regardless of num_items
        num_pairs = k * (k - 1) // 2
        pointer_cmp_size = num_pairs * 3
        data_cmp_size = num_pairs * 3 * num_attributes
        neighbour_size = k * 8 * num_attributes
        pointer_feat_size = k * self._num_pointer_features
        scalar_size = self._num_scalar_features

        self.observation_space = Dict(
            pointer_comparisons=MultiBinary(pointer_cmp_size),
            data_comparisons=MultiBinary(data_cmp_size),
            neighbour_comparisons=MultiBinary(neighbour_size),
            pointer_features=MultiBinary(pointer_feat_size),
            scalar_features=MultiBinary(scalar_size),
        )

    @property
    def _num_pointer_features(self):
        """Number of binary features per pointer (e.g. selected, fits)."""
        raise NotImplementedError

    @property
    def _num_scalar_features(self):
        """Number of global scalar binary features."""
        raise NotImplementedError

    def _generate_instance(self):
        """Generate a new problem instance. Must set self.items and self.num_items."""
        raise NotImplementedError

    def _get_pointer_features(self):
        """Return (k, _num_pointer_features) int8 array of per-pointer binary features."""
        raise NotImplementedError

    def _get_scalar_obs(self):
        """Return (_num_scalar_features,) int8 array of global binary features."""
        raise NotImplementedError

    def _get_item_attribute(self, item_idx, attr_idx):
        """Get attribute value for an item. Override for derived attributes (e.g. ratio)."""
        return self.items[item_idx][attr_idx]

    # ------------------------------------------------------------------
    # Observation building — constant-size, O(k² × num_attrs)
    # ------------------------------------------------------------------

    def _get_obs(self):
        k = self.k
        num_attrs = self.num_attributes
        num_pairs = k * (k - 1) // 2

        # Pointer position comparisons: v[i] < v[j], ==, >
        pointer_cmps = np.zeros(num_pairs * 3, dtype=np.int8)
        # Data comparisons per attribute
        data_cmps = np.zeros(num_pairs * 3 * num_attrs, dtype=np.int8)
        # Neighbour comparisons per attribute
        neighbour_cmps = np.zeros(k * 8 * num_attrs, dtype=np.int8)

        pair_idx = 0
        for i in range(k):
            for j in range(i + 1, k):
                base_ptr = pair_idx * 3
                pointer_cmps[base_ptr + 0] = self.v[i] < self.v[j]
                pointer_cmps[base_ptr + 1] = self.v[i] == self.v[j]
                pointer_cmps[base_ptr + 2] = self.v[i] > self.v[j]

                for a in range(num_attrs):
                    base_data = pair_idx * 3 * num_attrs + a * 3
                    val_i = self._get_item_attribute(self.v[i], a)
                    val_j = self._get_item_attribute(self.v[j], a)
                    data_cmps[base_data + 0] = val_i < val_j
                    data_cmps[base_data + 1] = val_i == val_j
                    data_cmps[base_data + 2] = val_i > val_j

                pair_idx += 1

        # Neighbour comparisons
        for i in range(k):
            for a in range(num_attrs):
                base = (i * num_attrs + a) * 8
                val = self._get_item_attribute(self.v[i], a)
                # Left neighbour
                if self.v[i] > 0:
                    left_val = self._get_item_attribute(self.v[i] - 1, a)
                    neighbour_cmps[base + 0] = 0  # not at boundary
                    neighbour_cmps[base + 1] = val > left_val
                    neighbour_cmps[base + 2] = val == left_val
                    neighbour_cmps[base + 3] = val < left_val
                else:
                    neighbour_cmps[base + 0] = 1  # at left boundary
                # Right neighbour
                if self.v[i] + 1 < self.num_items:
                    right_val = self._get_item_attribute(self.v[i] + 1, a)
                    neighbour_cmps[base + 4] = val > right_val
                    neighbour_cmps[base + 5] = val == right_val
                    neighbour_cmps[base + 6] = val < right_val
                    neighbour_cmps[base + 7] = 0  # not at boundary
                else:
                    neighbour_cmps[base + 7] = 1  # at right boundary

        pointer_feats = self._get_pointer_features().flatten()
        scalar_feats = self._get_scalar_obs()

        return OrderedDict([
            ('pointer_comparisons', pointer_cmps),
            ('data_comparisons', data_cmps),
            ('neighbour_comparisons', neighbour_cmps),
            ('pointer_features', pointer_feats),
            ('scalar_features', scalar_feats),
        ])

    # ------------------------------------------------------------------
    # Progressive difficulty
    # ------------------------------------------------------------------

    def _check_levelup(self):
        if len(self.reward_shortfalls) >= self.PROMOTION_WINDOW:
            recent = self.reward_shortfalls[-self.PROMOTION_WINDOW:]
            if min(recent) >= self.MIN_REWARD_SHORTFALL_FOR_PROMOTION:
                self.min_num_items += 1
                self.reward_shortfalls = []

    def _get_num_items(self):
        return self.np_random.integers(
            self.min_num_items,
            self.min_num_items + max(1, self.min_num_items // 2) + 1,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._check_levelup()
        self._generate_instance()
        # Reset pointers: even indices at 0, odd at end
        self.v[:] = 0
        self.v[::2] = 0
        self.v[1::2] = self.num_items - 1
        self.episode_total_reward = 0
        return self._get_obs(), {}

    def dispatch(self, instruction, args):
        self.instructions[instruction].implementation(args)

    def op_move_var(self, args):
        i, direction = args
        if direction:
            self.v[i] = min(self.v[i] + 1, self.num_items - 1)
        else:
            self.v[i] = max(self.v[i] - 1, 0)

    def op_assign_var(self, args):
        i, j = args
        self.v[i] = self.v[j]
