"""The instruction set as native tool-call schemas.

Constrained decoding makes malformed output impossible: the instruction name
and every argument range are enforced by the schema rather than by a parser and
a format penalty. Whether it helps the model *choose* better is a separate
question -- this only removes the format failure mode.
"""

from sorting_gym.text.parse import ParseError

SWAP_WITH_NEXT = "swap_with_next"
MOVE_VAR = "move_var"
ASSIGN_VAR = "assign_var"


def _pointer_schema(k, description):
    return {"type": "integer", "enum": list(range(k)), "description": description}


def tool_schemas(k):
    """OpenAI/ollama style function schemas for the three instructions."""
    return [
        {"type": "function", "function": {
            "name": SWAP_WITH_NEXT,
            "description": "Swap A[v_i] with the element immediately to its right.",
            "parameters": {"type": "object", "required": ["i"], "properties": {
                "i": _pointer_schema(k, "Pointer whose element is swapped with its right neighbour.")}}}},
        {"type": "function", "function": {
            "name": MOVE_VAR,
            "description": "Move pointer v_i one place left or right, stopping at the array ends.",
            "parameters": {"type": "object", "required": ["i", "direction"], "properties": {
                "i": _pointer_schema(k, "Pointer to move."),
                "direction": {"type": "string", "enum": ["+1", "-1"],
                              "description": "+1 moves right, -1 moves left."}}}}},
        {"type": "function", "function": {
            "name": ASSIGN_VAR,
            "description": "Set pointer v_i to the position of pointer v_j.",
            "parameters": {"type": "object", "required": ["i", "j"], "properties": {
                "i": _pointer_schema(k, "Pointer to overwrite."),
                "j": _pointer_schema(k, "Pointer whose position is copied.")}}}},
    ]


def _index(arguments, key, k):
    try:
        value = int(arguments[key])
    except (KeyError, TypeError, ValueError):
        raise ParseError(f"tool call missing integer argument {key!r}: {arguments!r}")
    if not 0 <= value < k:
        raise ParseError(f"pointer index {value} out of range for k={k}")
    return value


def action_from_tool_call(name, arguments, k):
    """Turn a tool call into the tuple the environment's `step` expects."""
    arguments = arguments or {}
    if name == SWAP_WITH_NEXT:
        return 0, _index(arguments, "i", k)
    if name == MOVE_VAR:
        direction = str(arguments.get("direction", "")).strip()
        if direction not in ("+1", "1", "-1"):
            raise ParseError(f"move_var direction must be +1 or -1, got {direction!r}")
        return 1, _index(arguments, "i", k), direction != "-1"
    if name == ASSIGN_VAR:
        return 2, _index(arguments, "i", k), _index(arguments, "j", k)
    raise ParseError(f"unknown tool {name!r}")
