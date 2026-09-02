# Which models can write a policy for this interface

Task: write `policy(facts) -> instruction string` for the array sorting env,
from scratch, scored on lengths 5/10/20 with 8 instances each (24 total).
20 independent candidates per model, one sample each, no refinement.

| model | size | fits 12GB | best of 20 | score spread across candidates |
|---|---|---|---|---|
| ornith-1.5:9b | 6.6 GB | yes | 1/24 | ten 0s, two 1s |
| qwen3:14b | 9.3 GB | yes | 1/24 | fourteen 0s, six 1s |
| gemma4:12b | 7.6 GB | yes | 8/24 | 0,1,2,3,5,8 — a real gradient |
| qwen3.6:27b | 17 GB | **no** | **24/24** | solved on its first candidate |

## The cliff

Nothing at or below 14B produced a correct policy in 20 attempts. The 27B
produced one on attempt one. The task is not large — the reference solutions in
`sorting_gym/agents/scripted.py` are ten-line decision trees over five
comparison bits — so this is a sharp capability threshold on a very small
program.

gemma4:12b is the only small model showing partial competence, and the shape of
its failure is informative: its better candidates solve the short instances and
then loop, with instruction mixes in the tens of thousands against a reference
of roughly 100. It writes policies that make local progress but never terminate.

## Consequence for hardware

The GPU cannot run the only model that works. qwen3.6:27b needs 17GB and the
card has 12, so it splits 47/53 across CPU and GPU and runs at 4 tok/s against
97 tok/s for a model that fits — roughly fifteen minutes per candidate.

That blocks fast iteration on everything downstream, because both the ring
control and the cost dose-response need a model that can solve the base task
before their results mean anything. Measured on models that cannot solve the
array either, both experiments returned nothing interpretable.

A 24GB card would fit the 27B and is also the floor veRL documents for its
smallest example (see `verl_notes.md`), so it unblocks both lines at once.

## Caveat

The 27B result is a single sample. It is the load-bearing data point for the
whole table and deserves repeating before anyone builds on it.
