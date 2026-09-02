"""Parse rendered observations and text actions back into structured form.

`parse_observation` exists to prove the rendered text is information
sufficient: an agent written against `ObservationFacts` sees exactly what a
scripted agent reading the raw bits would see, and nothing more.

`parse_action` is the inverse of a model's output: it turns `MoveVar(1, +1)`
into the `DiscreteParametric` tuple the environment's `step` expects.
"""

import re
from dataclasses import dataclass, field

from sorting_gym.text.render import BITS, NO_LEFT, NO_RIGHT, SEMANTIC


class ParseError(ValueError):
    """Raised when model output is not a well formed observation or action."""


_ACCESSORS = (
    "v_less_than(i, j)", "v_equals(i, j)", "v_greater_than(i, j)",
    "data_less_than(i, j)", "data_equals(i, j)", "data_greater_than(i, j)",
    "data_neighbour_greater(i, direction)", "data_neighbour_less(i, direction)",
    "at_left_edge(i)", "at_right_edge(i)",
)


@dataclass
class ObservationFacts:
    """The comparison predicates recovered from a rendered observation.

    Accessors mirror the ones in `sorting_gym.agents.scripted` so that a text
    agent reads like its bit-level counterpart.
    """
    k: int
    left: dict = field(default_factory=dict)
    right: dict = field(default_factory=dict)
    position: dict = field(default_factory=dict)
    value: dict = field(default_factory=dict)

    def __getattr__(self, name):
        # Only reached when normal lookup fails. Generated policies invent
        # plausible-sounding accessors, and this error text is fed back to the
        # model that wrote them, so it has to name what actually exists.
        raise AttributeError(
            f"ObservationFacts has no {name!r}. Available: {', '.join(_ACCESSORS)}")

    def _check(self, *indices):
        """Reject out-of-range pointers with a message naming the valid range.

        A bare KeyError here says nothing a caller can act on, which matters
        when the caller is a generated program whose error text is fed back to
        whatever produced it.
        """
        for index in indices:
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < self.k:
                raise ParseError(
                    f"pointer index {index!r} out of range; expected 0 to {self.k - 1}")

    def at_left_edge(self, i):
        self._check(i)
        return self.left[i] is None

    def at_right_edge(self, i):
        self._check(i)
        return self.right[i] is None

    def _pair(self, table, i, j):
        self._check(i, j)
        # A pointer compared with itself is trivially equal. Only pairs with
        # i < j are stored, so without this a caller asking about (i, i) --
        # which is a perfectly reasonable thing to ask -- gets a KeyError.
        if i == j:
            return "=="
        if i < j:
            return table[(i, j)]
        flipped = {"<": ">", ">": "<", "==": "=="}
        return flipped[table[(j, i)]]

    def v_less_than(self, i, j):
        return self._pair(self.position, i, j) == "<"

    def v_equals(self, i, j):
        return self._pair(self.position, i, j) == "=="

    def v_greater_than(self, i, j):
        return self._pair(self.position, i, j) == ">"

    def data_less_than(self, i, j):
        return self._pair(self.value, i, j) == "<"

    def data_equals(self, i, j):
        return self._pair(self.value, i, j) == "=="

    def data_greater_than(self, i, j):
        return self._pair(self.value, i, j) == ">"

    def data_neighbour_greater(self, i, direction):
        self._check(i)
        side = self.left if direction < 0 else self.right
        return side[i] == ">"

    def data_neighbour_less(self, i, direction):
        self._check(i)
        side = self.left if direction < 0 else self.right
        return side[i] == "<"


_NEIGHBOUR_LINE = re.compile(r"^v(\d+): (.+) \| (.+)$")
_PAIR_LINE = re.compile(
    r"^v(\d+) vs v(\d+): v\1 (<|==|>) v\2 \| A\[v\1\] (<|==|>) A\[v\2\]$")
_LEFT_CMP = re.compile(r"^A\[v(\d+)\] (<|==|>) A\[v\1-1\]$")
_RIGHT_CMP = re.compile(r"^A\[v(\d+)\] (<|==|>) A\[v\1\+1\]$")

_POSITION_SYMBOLS = ("<", "==", ">")
_VALUE_SYMBOLS = ("<", "==", ">")


def _parse_bits(text, k):
    facts = ObservationFacts(k=k)
    groups = {}
    for line in text.strip().splitlines():
        label, _, rest = line.partition(":")
        groups[label.strip()] = rest.split()
    if set(groups) != {"neighbours", "pairs"}:
        raise ParseError(f"Expected 'neighbours' and 'pairs' rows, got {sorted(groups)}")

    neighbours = groups["neighbours"]
    if len(neighbours) != k:
        raise ParseError(f"Expected {k} neighbour groups, got {len(neighbours)}")
    for i, group in enumerate(neighbours):
        if len(group) != 8:
            raise ParseError(f"Neighbour group {i} is not 8 bits: {group!r}")
        bits = [b == "1" for b in group]
        facts.left[i] = None if bits[0] else next(
            (s for s, b in zip(_VALUE_SYMBOLS, (bits[3], bits[2], bits[1])) if b), None)
        facts.right[i] = None if bits[7] else next(
            (s for s, b in zip(_VALUE_SYMBOLS, (bits[6], bits[5], bits[4])) if b), None)

    pairs = groups["pairs"]
    expected = [(i, j) for i in range(k) for j in range(i + 1, k)]
    if len(pairs) != len(expected):
        raise ParseError(f"Expected {len(expected)} pair groups, got {len(pairs)}")
    for (i, j), group in zip(expected, pairs):
        if len(group) != 6:
            raise ParseError(f"Pair group {(i, j)} is not 6 bits: {group!r}")
        bits = [b == "1" for b in group]
        facts.position[(i, j)] = next(
            (s for s, b in zip(_POSITION_SYMBOLS, bits[0:3]) if b), None)
        facts.value[(i, j)] = next(
            (s for s, b in zip(_VALUE_SYMBOLS, bits[3:6]) if b), None)
    return facts


def parse_observation(text, k, style=SEMANTIC):
    """Recover `ObservationFacts` from text produced by `render_observation`."""
    if style == BITS:
        return _parse_bits(text, k)
    if style != SEMANTIC:
        raise ValueError(f"Unknown style {style!r}")

    facts = ObservationFacts(k=k)
    for line in text.strip().splitlines():
        line = line.strip()
        pair_match = _PAIR_LINE.match(line)
        if pair_match:
            i, j = int(pair_match.group(1)), int(pair_match.group(2))
            facts.position[(i, j)] = pair_match.group(3)
            facts.value[(i, j)] = pair_match.group(4)
            continue

        neighbour_match = _NEIGHBOUR_LINE.match(line)
        if not neighbour_match:
            raise ParseError(f"Unrecognised observation line: {line!r}")
        i = int(neighbour_match.group(1))
        left, right = neighbour_match.group(2), neighbour_match.group(3)

        if left == NO_LEFT:
            facts.left[i] = None
        else:
            match = _LEFT_CMP.match(left)
            if not match or int(match.group(1)) != i:
                raise ParseError(f"Unrecognised left comparison: {left!r}")
            facts.left[i] = match.group(2)

        if right == NO_RIGHT:
            facts.right[i] = None
        else:
            match = _RIGHT_CMP.match(right)
            if not match or int(match.group(1)) != i:
                raise ParseError(f"Unrecognised right comparison: {right!r}")
            facts.right[i] = match.group(2)

    missing = [i for i in range(k) if i not in facts.left]
    if missing:
        raise ParseError(f"No neighbour line for pointer(s) {missing}")
    return facts


_ACTION = re.compile(r"^\s*(\w+)\s*\(([^)]*)\)\s*$")


def _pointer(token, k):
    try:
        index = int(token)
    except ValueError:
        raise ParseError(f"Expected a pointer index, got {token!r}")
    if not 0 <= index < k:
        raise ParseError(f"Pointer index {index} out of range for k={k}")
    return index


def parse_action(text, k):
    """Parse `SwapWithNext(i)`, `MoveVar(i, +1)` or `AssignVar(i, j)`.

    Returns the tuple accepted by the basic environment's `step`. Raises
    `ParseError` on anything malformed, which is what a caller turns into a
    format penalty.
    """
    match = _ACTION.match(text)
    if not match:
        raise ParseError(f"Not an instruction call: {text!r}")
    name = match.group(1)
    arguments = [a.strip() for a in match.group(2).split(",")] if match.group(2).strip() else []

    if name == "SwapWithNext":
        if len(arguments) != 1:
            raise ParseError(f"SwapWithNext takes 1 argument, got {len(arguments)}")
        return 0, _pointer(arguments[0], k)

    if name == "MoveVar":
        if len(arguments) != 2:
            raise ParseError(f"MoveVar takes 2 arguments, got {len(arguments)}")
        direction = arguments[1]
        if direction not in ("+1", "1", "-1"):
            raise ParseError(f"MoveVar direction must be +1 or -1, got {direction!r}")
        return 1, _pointer(arguments[0], k), direction != "-1"

    if name == "AssignVar":
        if len(arguments) != 2:
            raise ParseError(f"AssignVar takes 2 arguments, got {len(arguments)}")
        return 2, _pointer(arguments[0], k), _pointer(arguments[1], k)

    raise ParseError(f"Unknown instruction {name!r}")


def format_action(action):
    """Inverse of `parse_action`, for building few-shot examples."""
    instruction, *arguments = action
    if instruction == 0:
        return f"SwapWithNext({arguments[0]})"
    if instruction == 1:
        return f"MoveVar({arguments[0]}, {'+1' if arguments[1] else '-1'})"
    if instruction == 2:
        return f"AssignVar({arguments[0]}, {arguments[1]})"
    raise ValueError(f"Cannot format instruction {instruction}")
