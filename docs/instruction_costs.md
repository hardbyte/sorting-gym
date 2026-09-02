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
`SwapWithNext` is adjacent-only, so a policy that only ever swaps an inverted
adjacent pair performs **exactly one swap per inversion**. On the same 8
instances at n=20:

| agent | total | SwapWithNext | MoveVar | AssignVar |
|---|---|---|---|---|
| bubble | 2210 | 659 | 1447 | 104 |
| insertion | 1505 | 659 | 696 | 150 |
| synthesized | 2766 | 659 | 2003 | 104 |

All three spend **the same 659 swaps** (the mean inversion count is 82.4, and
8 x 82.4 = 659). Pricing swaps at 20 adds exactly 1565 to each of these totals,
so it cannot reorder *them*.

That is a claim about swap-minimal policies, not about every correct policy.
One swap per inversion is a lower bound, not an invariant: a policy may swap an
already ordered pair and undo it later, and `op_swap_with_next` clamps to the
last index, so a swap at the right edge exchanges an element with itself and is
still billed. Since `v1` starts at the right edge, any policy can prepend such
a no-op and still sort. Pricing swaps therefore does separate a wasteful policy
from a swap-minimal one; what it cannot do is separate bubble from insertion,
because both are swap-minimal. Both facts are pinned in
`tests/test_instruction_costs.py`.

"Expensive swaps therefore selection sort" still does not follow here, and
needs the functional environment's arbitrary `Swap(i, j)`.

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

### From scratch under different prices

The cleaner test, and the more informative answer. Same model, same prompt
structure, only the price table differing:

| cost model | program | mean cost | instruction mix |
|---|---|---|---|
| uniform | bubble sort | 133.4 | swap 319, move 951, assign 49 |
| expensive_move | bubble sort | 2040.1 | swap 319, move 951, assign 49 |

Different source text - the second detects end of pass with `v_equals(0, 1)`
rather than `at_right_edge(0)` - but the same algorithm, the same instruction
mix, and the same cost. **The price list did not change what was written.**

It should have. Under `expensive_move` an insertion-style policy is about half
the price (reference insertion 826.4 against bubble 1576.8), and the policy the
model produced costs 2040.1.

So on the evidence here the model's prior towards bubble sort dominates the
stated objective. That is evidence *for* the memorisation concern that motivated
this work, not against it: a search that reproduces the same textbook algorithm
whatever it is asked to optimise is recalling rather than searching.

These runs were CPU-only at roughly 15 minutes per generation, so each cost
model has a single usable sample. The matching instruction mixes are striking
but the sample is small; treat it as provisional and rerun on a GPU with
several samples per price list before relying on it.
