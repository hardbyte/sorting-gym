"""Scripted agents that see only the rendered text.

These are line-for-line ports of `bubble_sort_agent` and
`insertion_sort_agent` from `sorting_gym.agents.scripted`, but every predicate
comes from parsing the rendered observation and every decision is emitted as
an instruction string. If they sort as well as the originals, the text channel
carries everything the bit-level interface does -- which is the prerequisite
for putting a language model in this loop.
"""

from sorting_gym.text.parse import parse_action, parse_observation
from sorting_gym.text.render import SEMANTIC, render_observation


def bubble_sort_policy(facts):
    """c.f. Algorithm 2 - pg 19. Returns an instruction string."""
    i, j, el = 0, 1, 2
    if facts.v_less_than(i, j):
        if facts.data_neighbour_greater(i, +1):
            return f"SwapWithNext({i})"
        return f"MoveVar({i}, +1)"
    if facts.v_equals(i, j):
        return f"MoveVar({j}, -1)"
    return f"AssignVar({i}, {el})"


def insertion_sort_policy(facts):
    """c.f. Algorithm 4 - pg 20. Returns an instruction string."""
    i, j, low = 0, 1, 2
    if facts.v_less_than(i, j):
        return f"AssignVar({j}, {i})"
    if facts.v_equals(i, j):
        return f"MoveVar({i}, +1)"
    if facts.data_neighbour_greater(j, +1):
        return f"SwapWithNext({j})"
    if facts.v_greater_than(j, low) and facts.data_neighbour_less(j, -1):
        return f"MoveVar({j}, -1)"
    return f"AssignVar({j}, {i})"


def _text_agent(policy, style):
    def agent(observation, k):
        text = render_observation(observation, k, style=style)
        facts = parse_observation(text, k, style=style)
        return parse_action(policy(facts), k)
    return agent


def text_bubble_sort_agent(observation, k, style=SEMANTIC):
    return _text_agent(bubble_sort_policy, style)(observation, k)


def text_insertion_sort_agent(observation, k, style=SEMANTIC):
    return _text_agent(insertion_sort_policy, style)(observation, k)
