# Per-instruction costs

Instructions carry a price and the reward is the negative price, so the
objective can ask for a *different algorithm* rather than only a shorter
episode. Defaults are all `1.0`, reproducing the original `-1` per step.

```python
env = BasicNeuralSortInterfaceEnv(
    k=3, instruction_costs={"SwapWithNext": 1, "MoveVar": 1, "AssignVar": 20})
```

## Which cost axes can do anything

Measuring the reference agents first ruled out the obvious experiment.
`SwapWithNext` is adjacent-only, so **every correct policy performs exactly one
swap per inversion**. On the same 8 instances at n=20:

| agent | total | SwapWithNext | MoveVar | AssignVar |
|---|---|---|---|---|
| bubble | 2210 | 659 | 1447 | 104 |
| insertion | 1505 | 659 | 696 | 150 |
| synthesized | 2766 | 659 | 2003 | 104 |

All three spend **the same 659 swaps** (the mean inversion count is 82.4, and
8 x 82.4 = 659). Pricing swaps at 20 adds exactly 1565 to every total, so it
cannot change which algorithm wins. "Expensive swaps therefore selection sort"
needs the functional environment's arbitrary `Swap(i, j)`; it is a dead axis
here.

The live axis is `MoveVar` against `AssignVar`, which the two algorithms trade
in opposite directions. Bubble is cheaper than insertion once

    751 * cost(MoveVar) < 46 * cost(AssignVar)

i.e. once an assign costs more than about 16 moves. Measured episode cost at
n=20:

| cost model | bubble | insertion | cheaper |
|---|---|---|---|
| uniform | 276.2 | 188.1 | insertion |
| expensive_swap | 1841.4 | 1753.2 | insertion (same +1565 on both) |
| expensive_assign | 523.2 | 544.4 | **bubble** |
| expensive_move | 3712.9 | 1841.1 | insertion |

Both the invariance and the flip are pinned in `tests/test_instruction_costs.py`.

## Does discovery respond to the objective?

Open. The *scoring* demonstrably responds: the same two algorithms swap places
when only the price list changes. Whether an LLM search finds the cheaper
algorithm is not yet shown.

Seeding `examples/llm_synthesis.py` with the synthesized bubble policy and
running `ornith-1.5:9b` under `uniform` and `expensive_move` (6 generations
each, identical seed, only the prices differing) produced no movement: every
candidate either failed to sort or reproduced the seed verbatim. That is a
statement about a 9B model refining code on CPU, not about the hypothesis.

To settle it, run a larger model, and prefer `expensive_move`, where the
move-heavy seed costs 1904 against a 133 baseline and the pressure is largest.
