"""Render a neural-interface observation as text.

The renderer takes *only* the observation dict. It has no access to the
environment, so it cannot leak the array contents, the array length, or the
absolute pointer positions -- the omissions that make the paper's length
generalization claim work in the first place.
"""

from sorting_gym.agents.scripted import _get_pairwise_offset

SEMANTIC = "semantic"
BITS = "bits"

NO_LEFT = "no left neighbour"
NO_RIGHT = "no right neighbour"

# Bit offsets within a pointer's 8-bit neighbour row, as a (bit, comparison) pair.
_LEFT_BITS = ((1, ">"), (2, "=="), (3, "<"))
_RIGHT_BITS = ((4, ">"), (5, "=="), (6, "<"))
# Bit offsets within a pair's 6-bit group.
_POSITION_BITS = ((0, "<"), (1, "=="), (2, ">"))
_VALUE_BITS = ((3, "<"), (4, "=="), (5, ">"))


def _first_set(bits, base, candidates):
    for offset, symbol in candidates:
        if bits[base + offset]:
            return symbol
    return None


def _neighbour_line(neighbours, i):
    base = 8 * i
    if neighbours[base]:
        left = NO_LEFT
    else:
        comparison = _first_set(neighbours, base, _LEFT_BITS)
        left = NO_LEFT if comparison is None else f"A[v{i}] {comparison} A[v{i}-1]"

    if neighbours[base + 7]:
        right = NO_RIGHT
    else:
        comparison = _first_set(neighbours, base, _RIGHT_BITS)
        right = NO_RIGHT if comparison is None else f"A[v{i}] {comparison} A[v{i}+1]"

    return f"v{i}: {left} | {right}"


def _pair_line(pairwise, i, j, k):
    base = _get_pairwise_offset(i, j, k)
    position = _first_set(pairwise, base, _POSITION_BITS)
    value = _first_set(pairwise, base, _VALUE_BITS)
    return f"v{i} vs v{j}: v{i} {position} v{j} | A[v{i}] {value} A[v{j}]"


def render_observation(observation, k, style=SEMANTIC):
    """Render `observation` as text for a language model.

    `style` is SEMANTIC (readable, one line per pointer and per pair) or BITS
    (the raw groups, for a terse-versus-verbose ablation). Both round-trip
    through `sorting_gym.text.parse.parse_observation`.
    """
    neighbours = observation['neighbour_view_comparisons']
    pairwise = observation['pairwise_view_comparisons']

    if style == BITS:
        neighbour_groups = " ".join(
            "".join(str(int(b)) for b in neighbours[8 * i:8 * i + 8]) for i in range(k))
        pair_groups = " ".join(
            "".join(str(int(b)) for b in pairwise[_get_pairwise_offset(i, j, k):
                                                  _get_pairwise_offset(i, j, k) + 6])
            for i in range(k) for j in range(i + 1, k))
        return f"neighbours: {neighbour_groups}\npairs: {pair_groups}"

    if style != SEMANTIC:
        raise ValueError(f"Unknown style {style!r}")

    lines = [_neighbour_line(neighbours, i) for i in range(k)]
    lines += [_pair_line(pairwise, i, j, k) for i in range(k) for j in range(i + 1, k)]
    return "\n".join(lines)
