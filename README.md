# Sorting Gym

Gymnasium Environments for Sorting and Combinatorial Optimization, based on the 2020 paper
[_Strong Generalization and Efficiency in Neural Programs_](https://arxiv.org/abs/2007.03629) by
_Yujia Li, Felix Gimeno, Pushmeet Kohli, Oriol Vinyals_.

The key insight: RL agents manipulate **k pointer variables** over problem data, receiving only
**constant-size O(k²) pairwise comparison observations** — independent of problem size. This enables
strong generalization to larger instances than seen during training.

Install from pypi (recommended) with:
```
pip install sorting-gym
```

## Environments

### Sorting

- `SortTapeAlgorithmicEnv-v0` - Tape based environment that generates random sequences for sorting with progressive difficulty.
- `BasicNeuralSortInterfaceEnv-v0` - an interface where agents can implement simple algorithms such as bubble sort and insertion sort.
- `FunctionalNeuralSortInterfaceEnv-v0` - extends the `BasicNeuralSortInterfaceEnv-v0` interface to include instructions for entering and exiting functions.

#### Observation Space

The agent **never sees raw array values**. Instead, it receives constant-size binary comparison
features derived from **k pointer variables** (`v_1, ..., v_k`) over the array:

| Feature group | Size | Description |
|---|---|---|
| **Neighbour comparisons** | `8k` bits | For each pointer `v_i`: 4 bits for the left neighbour (`at_boundary`, `>`, `==`, `<`) and 4 bits for the right neighbour (`>`, `==`, `<`, `at_boundary`) |
| **Pairwise comparisons** | `6 × k(k-1)/2` bits | For each pointer pair `(i,j)`: position comparisons (`v_i < v_j`, `==`, `>`) and value comparisons (`A[v_i] < A[v_j]`, `==`, `>`) |

With the default `k=3`, this gives **42 binary features** — constant regardless of array length.
This is the key property enabling generalization to longer sequences than seen during training.

#### Action Space

Actions use a `DiscreteParametric` space — the agent first selects an instruction, then provides
instruction-specific arguments:

| Instruction | Arguments | Effect |
|---|---|---|
| `SwapWithNext(i)` | pointer index `i ∈ [0, k)` | Swap `A[v_i]` and `A[v_i + 1]` |
| `MoveVar(i, dir)` | pointer `i ∈ [0, k)`, direction `∈ {0, 1}` | Increment or decrement `v_i` (clamped to array bounds) |
| `AssignVar(i, j)` | pointers `i, j ∈ [0, k)` | Set `v_i = v_j` |

**Reward:** -1 per step. The episode terminates when the array is sorted.
**Progressive difficulty:** array length increases automatically as the agent's performance improves.

### Combinatorial Optimization

- `KnapsackEnv-v0` - 0/1 Knapsack problem. Items have weight, value, and value/weight ratio as comparison attributes. Agent selects items without exceeding capacity.
- `BinPackingEnv-v0` - 1D Bin Packing problem. Items have sizes and must be packed into minimum number of fixed-capacity bins.
- `JobShopSchedulingEnv-v0` - Job Shop Scheduling. Schedule operations (job × machine) to minimize makespan.

All combinatorial environments share the same pointer-based neural interface design:
- **Constant-size observations** via pairwise comparisons between pointer-referenced items
- **Instruction-based actions** (select, move pointer, assign pointer, finish)
- **Progressive difficulty** — instance size increases as the agent improves
- Scripted heuristic agents for validation (greedy knapsack, first-fit-decreasing, SPT)

## Parametric Action Space

To define the parametric action space we introduce the `DiscreteParametric(Space)` type,
allowing environments to describe disjoint output spaces, conditioned on a discrete parameter space.
For example:

```python
from gymnasium.spaces import Discrete, Tuple, MultiBinary
from sorting_gym import DiscreteParametric
action_space = DiscreteParametric(2, ([Discrete(2), Tuple([Discrete(3), MultiBinary(3)])]))
action_space.sample()
(1, 2, array([0, 1, 0], dtype=int8))
action_space.sample()
(0, 1)
```

For agents that don't support a parametric action space, we provide wrappers (`BoxActionSpaceWrapper`,
`MultiDiscreteActionSpaceWrapper`, `DisjointMultiDiscreteActionSpaceWrapper`) that flatten the
`DiscreteParametric` action space.

RL Agents may want to consider supporting parametric/auto-regressive actions:
- https://docs.ray.io/en/master/rllib-models.html#autoregressive-action-distributions
- https://arxiv.org/abs/1502.03509


### Goals:

- [x] Implement bubblesort/insertion sort environment.
- [x] Implement bubblesort/insertion sort agents as tests.
- [x] Implement function environment.
- [x] Implement quick sort scripted agent to test function environment.
- [x] Wrap the environment to expose a box action space.
- [x] Wrap the environment to expose a single MultiDiscrete action space.
- [x] Wrap the environment to expose a Parametric action space where each disjoint space is a
      MultiDiscrete action space. See `DisjointMultiDiscreteActionSpaceWrapper`
- [x] 0/1 Knapsack environment with greedy scripted agent
- [x] 1D Bin Packing environment with first-fit-decreasing scripted agent
- [x] Job Shop Scheduling environment with SPT scripted agent
- [x] Include an example solution to train an agent via RL
- [ ] Environment rendering (at least text based, optional dependency for rendering graphically with e.g. pygame)


### Ideas to take it further:

- Accelerate environment with cython (if required)
- Abstract out a Neural Controller Mixin/Environment Wrapper?
- Consider a different/enhanced instruction set.
  Instead of always comparing every pointer and data element in the view (and neighbours),
  have explicit comparison instructions. Could extend to other math instructions, including
  accounting for variable cost of the instructions.
- Instead of passing previous arguments, consider passing in the number of instructions
  executed in the current scope as a cheap program counter.
- Add more combinatorial optimization problems (TSP, graph coloring, etc.)


## Training an RL Agent

An example PPO training script is provided using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/):

```bash
pip install stable-baselines3
python examples/train_ppo.py --env sort --timesteps 200000
```

The `MultiDiscreteActionSpaceWrapper` flattens the `DiscreteParametric` action space into a single
`MultiDiscrete` so standard RL libraries can consume it. `FlattenObservation` flattens the Dict
observation into a 1-D vector.

With a 256×256 MLP and 200k timesteps the sorting agent reaches near-optimal performance on small
arrays (reward of -1, meaning sorted in a single step on many episodes).

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```
uv sync
uv run pytest
```

## Building/Packaging

```
uv build
```
