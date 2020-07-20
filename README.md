# Sorting Gym

OpenAI Gym Environments for Sorting based on the 2020 paper
[_Strong Generalization and Efficiency in Neural Programs_](https://arxiv.org/abs/2007.03629) by 
_Yujia Li, Felix Gimeno, Pushmeet Kohli, Oriol Vinyals_.

This repository includes implementations of the basic neural environment for sorting.

Install from pypi (recommended) with:
```
pip install sorting-gym
```

Importing the Python package `sorting_gym` will expose the following Gym environments:

- `SortTapeAlgorithmicEnv-v0`
- `BasicNeuralSortInterfaceEnv-v0`

To define the parametric action space we introduce the `DiscreteParametric(Space)` type,
allowing environments to describe disjoint output spaces, conditioned on a discrete space.
For example:

```python
from gym.spaces import Discrete
from sorting_gym import DiscreteParametric
action_space = DiscreteParametric(2, ([Discrete(2), Discrete(3)]))
```

In the `agents` module we implement the scripted agents from the paper.

RL Agents may want to consider supporting parametric/auto-regressive actions:
- https://docs.ray.io/en/master/rllib-models.html#autoregressive-action-distributions
- https://arxiv.org/abs/1502.03509


### Goals:

- [x] Implement bubblesort/insertion sort environment.
- [x] Implement bubblesort/insertion sort agents as tests.
- [ ] Implement function stack environment
- [ ] Implement quick sort agent to test function environment
- [ ] Include an example solution to train an agent via RL
- [ ] Environment rendering

### Ideas to take it further:

- [ ] Accelerate environment with cython (if required)
- [ ] Open PR to `gym` for a discrete parametric space
- [ ] Abstract out a Neural Controller Mixin/Environment Wrapper?
- [ ] Consider a different/enhanced instruction set. 
      Instead of always comparing every pointer and data element in the view (and neighbours), 
      have explicit comparison instructions. Could extend to other math instructions, including
      accounting for variable cost of the instructions.
  

## Run test with pytest

```
pytest
```

## Building/Packaging

```
poetry update
poetry build
poetry package
```