"""Prompts for driving the neural sorting interface with a language model.

Nothing here may mention the array's contents or its length: the model is meant
to work from comparison facts alone, which is what makes the interface
length-agnostic in the first place.
"""

import numpy as np

from sorting_gym.text.agents import insertion_sort_policy
from sorting_gym.text.parse import parse_observation
from sorting_gym.text.render import BITS, SEMANTIC, render_observation

_INSTRUCTIONS = """\
  SwapWithNext(i)   swap A[v_i] with A[v_i + 1]
  MoveVar(i, +1)    move pointer v_i one place right (stops at the end)
  MoveVar(i, -1)    move pointer v_i one place left (stops at the start)
  AssignVar(i, j)   set v_i = v_j"""

_SEMANTIC_FORMAT = """\
Each turn you are given one line per pointer and one line per pair of pointers:

  v0: no left neighbour | A[v0] > A[v0+1]
  v1 vs v2: v1 < v2 | A[v1] == A[v2]

"no left neighbour" means the pointer is at the start of the array; "no right
neighbour" means it is at the end."""

_BITS_FORMAT = """\
Each turn you are given the raw comparison bits.

"neighbours" has one 8-bit group per pointer v_i, in order:
  at_start, A[v_i]>A[v_i-1], A[v_i]==A[v_i-1], A[v_i]<A[v_i-1],
  A[v_i]>A[v_i+1], A[v_i]==A[v_i+1], A[v_i]<A[v_i+1], at_end

"pairs" has one 6-bit group per pointer pair (i,j) with i<j, in order:
  v_i<v_j, v_i==v_j, v_i>v_j, A[v_i]<A[v_j], A[v_i]==A[v_j], A[v_i]>A[v_j]"""


def system_prompt(k, style=SEMANTIC, chain_of_thought=False, tool_calling=False):
    fmt = _BITS_FORMAT if style == BITS else _SEMANTIC_FORMAT
    pointers = ", ".join(f"v{i}" for i in range(k))
    if tool_calling:
        closing = "Call exactly one of the provided tools each turn."
    elif chain_of_thought:
        closing = ("Think step by step in at most two short sentences, then give the "
                   "instruction alone on the final line.")
    else:
        closing = "Reply with the instruction and nothing else."
    return f"""\
You are sorting an array A of integers into ascending order.

You cannot see the values in A, and you cannot see how long it is. You have \
{k} pointers into the array ({pointers}). All you ever observe is the result of \
comparisons between the elements the pointers refer to, and between the pointer \
positions themselves.

Each turn you must reply with exactly one instruction:
{_INSTRUCTIONS}
where i and j are pointer indices from 0 to {k - 1}.

{fmt}

At the start v0 and v2 are at the first element and v1 is at the last.
The episode ends as soon as A is sorted. Use as few instructions as you can.

{closing}"""


def sample_demonstrations(k, count, style=SEMANTIC, seed=0, base=10):
    """(rendered observation, expert instruction) pairs for few-shot prompting.

    Expert actions come from `insertion_sort_policy`, so the demonstrations show
    a policy that is known to solve the task through this exact text channel.
    """
    from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv

    rng = np.random.default_rng(seed)
    env = BasicNeuralSortInterfaceEnv(k=k, base=base)
    env.reset(seed=seed)

    demonstrations, seen = [], set()
    while len(demonstrations) < count:
        length = int(rng.integers(4, 9))
        env.A = list(rng.integers(0, base, length))
        env.v[:] = rng.integers(0, length, k)
        observation = env._get_obs()
        text = render_observation(observation, k, style=style)
        if text in seen:
            continue
        seen.add(text)
        action = insertion_sort_policy(parse_observation(text, k, style=style))
        demonstrations.append((text, action))
    return demonstrations


def build_messages(observation_text, k, style=SEMANTIC, demonstrations=(),
                   chain_of_thought=False, tool_calling=False):
    """Chat messages for one turn. Stateless: the observation is Markov."""
    messages = [{"role": "system",
                 "content": system_prompt(k, style=style,
                                          chain_of_thought=chain_of_thought,
                                          tool_calling=tool_calling)}]
    for demo_text, demo_action in demonstrations:
        messages.append({"role": "user", "content": demo_text})
        messages.append({"role": "assistant", "content": demo_action})
    messages.append({"role": "user", "content": observation_text})
    return messages
