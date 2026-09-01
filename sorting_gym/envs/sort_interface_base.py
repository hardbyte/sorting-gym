import numpy as np
from gymnasium import Env

from dataclasses import replace

from sorting_gym import DiscreteParametric
from sorting_gym.envs.tape import SortTapeAlgorithmicEnv


def _apply_costs(instructions, instruction_costs):
    """Override instruction costs by name.

    Costs are what the agent is actually optimising, so making them settable
    lets the same environment ask for different algorithms. Note that
    SwapWithNext is adjacent-only, so any correct policy performs exactly one
    swap per inversion: pricing swaps shifts every policy's total by the same
    amount and cannot change which algorithm wins. The live axis is MoveVar
    against AssignVar.
    """
    if not instruction_costs:
        return instructions
    known = {instruction.name for instruction in instructions}
    unknown = set(instruction_costs) - known
    if unknown:
        raise ValueError(f"unknown instruction(s) {sorted(unknown)}; expected {sorted(known)}")
    return [replace(instruction, cost=float(instruction_costs.get(instruction.name,
                                                                 instruction.cost)))
            for instruction in instructions]


class NeuralSortInterfaceEnv(Env):
    """
    Base for Neural interface based environment

    Keeps track of k index variables in `self.v`
    Generates data on environment reset, exposed as `self.A`

    Concrete implementations must implement `step` and create an observation.
    """

    def __init__(self, base, k, instructions, max_episode_steps=None,
                 allow_sorted_instances=False, instruction_costs=None):
        self.base = base
        self.k = k
        self.instructions = _apply_costs(instructions, instruction_costs)
        self.v = np.zeros(shape=k, dtype=np.int32)
        self.A = None
        # None means "scale the budget with the instance": an O(n^2) sort needs
        # roughly 0.5*n^2 instructions, so 4*n^2 leaves plenty of headroom.
        self.max_episode_steps = max_episode_steps
        self.steps_taken = 0
        self.episode_cost = 0.0
        # Generates random data for each episode increasing the length as the agent "levels up"
        self.tape_env = SortTapeAlgorithmicEnv(
            base=base, starting_min_length=4, allow_sorted_instances=allow_sorted_instances)

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
        self.episode_cost = 0.0

    @property
    def step_budget(self):
        if self.max_episode_steps is not None:
            return self.max_episode_steps
        return 4 * len(self.A) ** 2

    def instruction_cost(self, instruction):
        return self.instructions[instruction].cost

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
