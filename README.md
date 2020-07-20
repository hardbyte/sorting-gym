# Sorting Gym

OpenAI Gym Environments for Sorting based on the 2020 paper
[_Strong Generalization and Efficiency in Neural Programs_](https://arxiv.org/abs/2007.03629) by 
_Yujia Li, Felix Gimeno, Pushmeet Kohli, Oriol Vinyals_.

This repository includes implementations of the basic neural environment for sorting.

Install from pypi (recommended) with:
```
pip install sortingenv
```

Environments:

- `SortTapeAlgorithmicEnv-v0`
- `BasicNeuralSortInterfaceEnv-v0`

In the tests module we implement the manual agents from the paper.

Agents may want to consider supporting parametric/auto-regressive actions:
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

- [ ] Open PR to `gym` for a discrete parametric space
- [ ] Abstract out a Neural Controller Mixin/Environment Wrapper?
- [ ] Consider a different/enhanced instruction set.

## Building/Packaging

```
poetry build
poetry package
```