# What is expressible over O(k²) comparison bits

*Are Graph Neural Networks Optimal Approximation Algorithms?* (Yau et al.,
NeurIPS 2024) proves what a GNN architecture can **represent** — polynomial
message-passing captures the best polynomial-time algorithms for Max-CSPs under
the Unique Games Conjecture. The equivalent question for this repo is smaller
and answerable by measurement:

> Which sorting algorithms are expressible as a **memoryless** function of the
> constant-size observation, and how many pointers does each need?

"Memoryless" matters. The basic environment's observation is exactly
`neighbour_view_comparisons` and `pairwise_view_comparisons` — no history — so
a policy there is a pure function from O(k²) bits to one instruction. Anything
needing state must either reconstruct it from pointer positions or move to the
functional environment, whose observation adds `current_function` and
`previous_action`.

## Measured

15 random instances of length 12 per cell, `max_episode_steps = 60n²`. `env:`
means the environment cannot be built at that k; a bare exception name means the
agent asked for a pointer that does not exist.

### Array environment

| agent | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| gnome | env:AssertionError | **15/15** | 15/15 | 15/15 |
| bubble | env:AssertionError | IndexError | **15/15** | 15/15 |
| insertion | env:AssertionError | IndexError | **15/15** | 15/15 |
| ring_seam | env:AssertionError | 0/15 | 0/15 | 0/15 |

### Ring environment (no edges, goal is any rotation of sorted)

| agent | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| gnome | env:AssertionError | **9/15** | 9/15 | 9/15 |
| bubble | env:AssertionError | IndexError | **15/15** | 15/15 |
| insertion | env:AssertionError | IndexError | **15/15** | 15/15 |
| ring_seam | env:AssertionError | **15/15** | 15/15 | 15/15 |

## What the table says

**Gnome sort needs one pointer's worth of information.** It reads only the 8
neighbour bits of v0 and ignores every pairwise bit, so it is the cheapest
sorting algorithm in this interface. It is also the only one of the three array
sorts that never consults another pointer.

**k=1 is unreachable for a structural reason, not an algorithmic one.** With one
pointer the pairwise component has size `6·k(k−1)/2 = 0`, and Gymnasium's
`MultiBinary` requires n > 0, so the environment cannot be constructed. Gnome
would run there unchanged. Testing the k=1 claim needs the observation to admit
an empty pairwise component.

**Bubble and insertion need k=3 as written**, and fail at k=2 by naming pointer
2 — bubble uses it as a fixed anchor at the start of the array, insertion as the
low watermark. This is a fact about these implementations, not a proof about the
algorithms: whether *some* memoryless bubble sort exists at k=2 is open.

**Both paper agents survive losing the edges; the synthesized one does not.**
Bubble and insertion each solve the ring 15/15 unchanged, because they anchor on
pointer comparisons (`v_less_than`, `v_equals`) which still work when there is
no first element. The policy an LLM wrote for the array anchors on
`at_left_edge` / `at_right_edge` and scores 0/20 on the ring (see
`model_capability.md`). The three are behaviourally identical on the array and
only the ring separates them.

**Gnome degrades on the ring rather than failing outright** (9/15). Without a
left boundary to stop at, its walk can circulate indefinitely — the same reason
cyclic bubble sort does not converge on a ring at all (0/30 for n ≥ 10, measured
while building `ring.py`). Termination on a ring appears to need a positional
anchor, which is what `ring_sort_agent` supplies by parking v1 on an arbitrary
seam.

**Quicksort needs memory, not more pointers.** `quicksort_agent` reads
`previous_action` in 22 places, and that key exists only in the functional
environment's observation. It is therefore not expressible in the basic
environment at any k without reconstructing history from pointer positions.

## Open

- Is a memoryless bubble or insertion sort expressible at k=2? A negative would
  need an argument over reachable observations, not a search over policies.
- The reachable observation set. The full space is 2^42 for k=3 (≈4.4×10¹², not
  enumerable), but one-hot comparison groups and total-order consistency
  constrain it heavily. If the reachable set is small enough to enumerate, a
  policy can be verified exhaustively rather than sampled — which would turn
  "solved 45/45 instances" into "provably sorts every array up to length n".
- Does the seam requirement generalise? The ring evidence suggests termination
  needs a positional anchor, but that is an observation across four agents, not
  a theorem.
- Cost changes the ranking but not expressibility. `instruction_costs.md`
  records which algorithm is *cheapest* under a price list; this document is
  about which are *possible* at all.
