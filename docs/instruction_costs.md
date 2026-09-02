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

### What was tried

Seeding `examples/llm_synthesis.py` with the synthesized bubble policy and
varying only the price list:

| model | cost model | generations | outcome |
|---|---|---|---|
| ornith-1.5:9b | uniform | 6 | broke, or reproduced the seed |
| ornith-1.5:9b | expensive_move | 6 | broke, or reproduced the seed |
| qwen3.6:27b | expensive_move | 6 | 2 reproduced the seed byte-identically, 4 unusable |

**Seeding looks counterproductive.** A prompt containing a working program
invites copying it: every usable candidate came back with the seed's exact cost
of 2040.1 and its exact instruction mix. The same model, asked to write a
policy *from scratch*, produced a correct one on its first attempt.

So the better test is from scratch under different prices, comparing the
programs that come back, rather than refinement from a shared starting point.

These runs were CPU-only, at roughly 15 minutes per generation, which is why
the sample is small. Treat the negative result as provisional.
