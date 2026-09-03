"""Optimal values and bounds for the combinatorial environments.

Without these a score is uninterpretable. "The policy collected value 47" says
nothing unless you know whether the best possible was 48 or 480, so these turn
raw returns into approximation ratios and give the environments a ceiling to
measure against.

Everything here reads the instance directly. That is fine for scoring, which
sits outside the agent's constant-size observation, but it means none of it may
be exposed to a policy.
"""

EXACT_ITEM_LIMIT = 18


def knapsack_optimal(items, capacity):
    """Exact 0/1 knapsack value by dynamic programming, O(n * capacity)."""
    best = [0] * (capacity + 1)
    for weight, value in items:
        if weight > capacity:
            continue
        for remaining in range(capacity, weight - 1, -1):
            candidate = best[remaining - weight] + value
            if candidate > best[remaining]:
                best[remaining] = candidate
    return best[capacity]


def knapsack_lp_bound(items, capacity):
    """Fractional relaxation: an upper bound, and never below the 0/1 optimum.

    Cheap enough for instances where the DP table would be large.
    """
    remaining = capacity
    total = 0.0
    for weight, value in sorted(items, key=lambda it: it[1] / it[0], reverse=True):
        if weight <= remaining:
            remaining -= weight
            total += value
        else:
            total += value * (remaining / weight)
            break
    return total


def bin_packing_lower_bound(sizes, bin_capacity):
    """Every item must go somewhere, so total size over capacity, rounded up."""
    if not sizes:
        return 0
    return -(-sum(sizes) // bin_capacity)


def bin_packing_optimal(sizes, bin_capacity):
    """Fewest bins, by branch and bound. Exact but exponential.

    Bin packing is NP-hard, so this refuses instances large enough to hang
    rather than quietly taking forever.
    """
    if len(sizes) > EXACT_ITEM_LIMIT:
        raise ValueError(
            f"exact bin packing is exponential; {len(sizes)} items exceeds the "
            f"{EXACT_ITEM_LIMIT} item limit. Use bin_packing_lower_bound instead.")
    if any(size > bin_capacity for size in sizes):
        raise ValueError("an item is larger than a bin")
    if not sizes:
        return 0

    ordered = sorted(sizes, reverse=True)
    lower = bin_packing_lower_bound(ordered, bin_capacity)
    best = [len(ordered)]

    def place(index, bins):
        if len(bins) >= best[0]:
            return
        if index == len(ordered):
            best[0] = len(bins)
            return
        size = ordered[index]
        seen = set()
        for i, remaining in enumerate(bins):
            if remaining >= size and remaining not in seen:
                seen.add(remaining)          # bins with equal room are interchangeable
                bins[i] -= size
                place(index + 1, bins)
                bins[i] += size
        bins.append(bin_capacity - size)     # or start a fresh bin
        place(index + 1, bins)
        bins.pop()

    place(0, [])
    return max(best[0], lower)


def approximation_ratio(achieved, optimal, sense="max"):
    """How close a result is to optimal, as a number in [0, 1].

    Reported the same way for both senses: 1.0 is optimal, lower is worse, so a
    maximisation ratio is achieved/optimal and a minimisation ratio is
    optimal/achieved.
    """
    if sense == "max":
        if optimal <= 0:
            return 1.0 if achieved >= optimal else 0.0
        return achieved / optimal
    if sense == "min":
        if achieved <= 0:
            return 1.0 if optimal <= 0 else 0.0
        return optimal / achieved
    raise ValueError(f"sense must be 'max' or 'min', got {sense!r}")
