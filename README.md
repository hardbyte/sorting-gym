# Sorting Gym

OpenAI Gym Environments for Sorting based on the 2020 paper
[_Strong Generalization and Efficiency in Neural Programs_](https://arxiv.org/abs/2007.03629) by 
_Yujia Li, Felix Gimeno, Pushmeet Kohli, Oriol Vinyals_.

This repository includes implementations of the neural interface environments for sorting.

Install from pypi (recommended) with:
```
pip install sorting-gym
```

Importing the Python package `sorting_gym` will expose the following Gym environments:

- `SortTapeAlgorithmicEnv-v0` - Tape based environment based on [Gym's algorithmic environment](https://github.com/openai/gym/blob/master/gym/envs/algorithmic/algorithmic_env.py#L242))
- `BasicNeuralSortInterfaceEnv-v0` - an interface where agents can implement simple algorithms such as bubble sort and insertion sort.
- `FunctionalNeuralSortInterfaceEnv-v0` - extends the `BasicNeuralSortInterfaceEnv-v0` interface to include instructions for entering and exiting functions.

To define the parametric action space we introduce the `DiscreteParametric(Space)` type,
allowing environments to describe disjoint output spaces, conditioned on a discrete parameter space.
For example:

```python
from gym.spaces import Discrete, Tuple, MultiBinary
from sorting_gym import DiscreteParametric
action_space = DiscreteParametric(2, ([Discrete(2), Tuple([Discrete(3), MultiBinary(3)])]))
action_space.sample()
(1, 2, array([0, 1, 0], dtype=int8))
action_space.sample()
(0, 1)
```

For agents that don't support a parametric action space, we provide two wrappers (`BoxActionSpaceWrapper` and 
`MultiDiscreteActionSpaceWrapper`) that flatten the `DiscreteParametric` action space down to a `Box` and a 
`MultiDiscrete` respectively. 

In the `sorting_gym.agents.scripted` module we implement the scripted agents from the paper directly using the 
unwrapped environment.

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
- [ ] Wrap the environment to expose a Parametric action space where each disjoint space is a
      MultiDiscrete action space. WIP in `DisjointMultiDiscreteActionSpaceWrapper`
- [ ] Include an example solution to train an agent via RL
- [ ] Environment rendering (at least text based, optional dependency for rendering graphically with e.g. pygame)
- [ ] Remove the open ai gym tape environment from base env (used to generate longer data as agent levels up)


### Ideas to take it further:

- Accelerate environment with cython (if required)
- Open PR to `gym` for a discrete parametric space
- Abstract out a Neural Controller Mixin/Environment Wrapper?
- Consider a different/enhanced instruction set. 
  Instead of always comparing every pointer and data element in the view (and neighbours), 
  have explicit comparison instructions. Could extend to other math instructions, including
  accounting for variable cost of the instructions.
- Instead of passing previous arguments, consider passing in the number of instructions
  executed in the current scope as a cheap program counter.


## Run test with pytest

```
pytest
```

## Building/Packaging

```
poetry update
poetry version patch
poetry lock
poetry build
poetry publish
```
