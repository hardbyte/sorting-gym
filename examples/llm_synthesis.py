"""Have a language model write the policy, instead of being the policy.

The reactive setup spends one generation per environment step, which is a lot
of inference for a task whose reference solutions are ten line decision trees
(`sorting_gym.agents.scripted`). Here the model is asked for the decision tree
itself: it writes a Python function over the observation facts, the environment
scores it on real instances, and the failures are fed back for another attempt.

Cost is per candidate program rather than per step, so a whole run is a few
dozen generations rather than tens of thousands. It also plays to what these
models are actually good at.

The candidate programs are executed locally. They come from a model, not a
trusted source, so they run with builtins restricted to a small allowlist --
enough to stop an accident, not a determined adversary.
"""

import argparse
import json
import re
import sys
import time
import urllib.request

import numpy as np

from sorting_gym.agents.scripted import (
    bubble_sort_agent, insertion_sort_agent, ring_sort_agent)
from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv
from sorting_gym.envs.ring import RingSortInterfaceEnv, is_rotation_of_sorted
from sorting_gym.text import ParseError, parse_action, parse_observation, render_observation

OLLAMA_URL = "http://localhost:11434/api/chat"

COST_MODELS = {
    "uniform": {"SwapWithNext": 1, "MoveVar": 1, "AssignVar": 1},
    "expensive_assign": {"SwapWithNext": 1, "MoveVar": 1, "AssignVar": 20},
    "expensive_move": {"SwapWithNext": 1, "MoveVar": 20, "AssignVar": 1},
    # Kept to show it is a dead axis: SwapWithNext is adjacent-only, so every
    # correct policy swaps once per inversion and this only offsets the total.
    "expensive_swap": {"SwapWithNext": 20, "MoveVar": 1, "AssignVar": 1},
}
CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
OPEN_BLOCK = re.compile(r"```(?:python)?\s*(.*)", re.DOTALL)

SAFE_BUILTINS = {
    "True": True, "False": False, "None": None,
    "abs": abs, "all": all, "any": any, "bool": bool, "int": int, "len": len,
    "max": max, "min": min, "range": range, "sorted": sorted, "sum": sum,
}

API = '''\
You write a policy for a sorting machine.

An array A of integers is sorted in place by moving k=3 pointers v0, v1, v2 and
swapping adjacent elements. You never see the values in A or its length. You
only see comparison facts, given to your function as `facts`, with these methods:

  facts.v_less_than(i, j)      -> bool   position of v_i is left of v_j
  facts.v_equals(i, j)         -> bool   v_i and v_j are at the same position
  facts.v_greater_than(i, j)   -> bool   v_i is right of v_j
  facts.data_less_than(i, j)   -> bool   A[v_i] <  A[v_j]
  facts.data_equals(i, j)      -> bool   A[v_i] == A[v_j]
  facts.data_greater_than(i, j)-> bool   A[v_i] >  A[v_j]
  facts.data_neighbour_greater(i, d) -> bool  A[v_i] > A[v_i + d], d is +1 or -1
  facts.data_neighbour_less(i, d)    -> bool  A[v_i] < A[v_i + d]
{edge_accessors}
Return exactly one instruction as a string:

  "SwapWithNext(i)"   swap A[v_i] with A[v_i + 1]
  "MoveVar(i, +1)"    move v_i one place right (stops at the end)
  "MoveVar(i, -1)"    move v_i one place left (stops at the start)
  "AssignVar(i, j)"   set v_i = v_j

{topology}

You cannot recover absolute positions or element values, and there is no way to
learn how long the array is. The comparisons listed above are the only
information available, so a policy that tries to reconstruct indices or the
contents of A cannot work. Decide purely from the relations.

{start_and_goal} Your
function is called fresh each step and keeps no state between calls, so every
decision must come from `facts` alone.

Each instruction has a price, and you are minimising the total price of an
episode, not the number of instructions:

{cost_table}

Prefer whichever instructions are cheap under these prices, as long as the
array still ends up sorted.

Write one function and nothing else:

```python
def policy(facts):
    ...
    return "MoveVar(0, +1)"
```'''


ARRAY_EDGES = """\
  facts.at_left_edge(i)        -> bool   v_i is at the first element
  facts.at_right_edge(i)       -> bool   v_i is at the last element
"""

ARRAY_TOPOLOGY = """\
MoveVar stops at the two ends of the array, and SwapWithNext at the last
element does nothing."""

ARRAY_START = """\
At the start of an episode v0 and v2 are at the first element, v1 at the last.
The episode ends as soon as A is sorted."""

RING_EDGES = ""

RING_TOPOLOGY = """\
A is arranged in a RING. There is no first or last element: moving right from
any element eventually returns to where you started, and SwapWithNext at any
element swaps it with the one after it around the ring. Nothing is at an edge,
because there are no edges."""

RING_START = """\
At the start of an episode v0 and v2 are together somewhere on the ring and v1
is elsewhere. The episode ends as soon as the ring reads in non-decreasing
order from some starting point - since the ring has no beginning, any rotation
of sorted order counts as sorted."""

ENVIRONMENTS = {
    "array": (BasicNeuralSortInterfaceEnv, ARRAY_EDGES, ARRAY_TOPOLOGY, ARRAY_START),
    "ring": (RingSortInterfaceEnv, RING_EDGES, RING_TOPOLOGY, RING_START),
}


def cost_table(costs):
    return "\n".join(f"  {name:14s} costs {value}"
                      for name, value in sorted(costs.items(), key=lambda kv: -kv[1]))


def build_api(costs, env_kind="array"):
    _cls, edges, topology, start = ENVIRONMENTS[env_kind]
    return (API.replace("{cost_table}", cost_table(costs))
               .replace("{edge_accessors}", edges)
               .replace("{topology}", topology)
               .replace("{start_and_goal}", start))


def call(model, messages, timeout=3600, think=False, max_tokens=4000, context=16384):
    # ollama defaults to a 4096 token context. The prompt plus a seed program
    # nearly fills that, leaving no room to generate, which shows up as a reply
    # containing no function at all rather than as an error.
    payload = json.dumps({
        "model": model, "messages": messages, "stream": False, "think": think,
        "options": {"temperature": 0.7, "num_predict": max_tokens,
                    "num_ctx": context},
    }).encode()
    request = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["message"]["content"]


def _candidate_sources(reply):
    """Every plausible way the function might be embedded in a reply.

    Models narrate around the code, and a long reply can be cut off before its
    closing fence. Falling back to the whole reply then tries to compile prose,
    or the fence marker itself, so an unterminated block and a bare `def` both
    need handling.
    """
    for block in CODE_BLOCK.findall(reply):
        if "def policy" in block:
            yield block
    open_block = OPEN_BLOCK.search(reply)
    if open_block and "def policy" in open_block.group(1):
        yield open_block.group(1)
    if "def policy" in reply:
        # Last resort: take the definition itself, dropping surrounding prose.
        yield reply[reply.index("def policy"):]


def extract_policy(reply):
    """Compile the model's reply into a callable, or raise ValueError."""
    source, errors = None, []
    for candidate in _candidate_sources(reply):
        try:
            compile(candidate, "<candidate>", "exec")
        except SyntaxError as error:
            errors.append(str(error))
            continue
        source = candidate
        break
    if source is None:
        if errors:
            raise ValueError(f"no compilable `def policy`: {errors[0]}")
        raise ValueError("no `def policy` in the reply")
    namespace = {"__builtins__": SAFE_BUILTINS}
    try:
        exec(compile(source, "<candidate>", "exec"), namespace)
    except Exception as error:                      # noqa: BLE001 - candidate code
        raise ValueError(f"candidate failed to compile: {error!r}") from error
    policy = namespace.get("policy")
    if not callable(policy):
        raise ValueError("`policy` is not callable")
    return policy, source


def _already_solved(array, env_kind):
    return is_rotation_of_sorted(array) if env_kind == "ring" else array == sorted(array)


def _unsolved_instance(rng, length, env_kind, base=10):
    """An instance that still needs work, so a do-nothing policy scores zero."""
    while True:
        array = list(rng.integers(0, base, length))
        if not _already_solved(array, env_kind):
            return array


def evaluate(policy, k=3, lengths=(5, 10, 20), instances=20, seed=0, budget_factor=4,
             costs=None, env_kind="array"):
    """Solve rate and step counts on real instances, plus the first error seen."""
    rng = np.random.default_rng(seed)
    solved, steps, costs_seen, failures, first_error = 0, [], [], [], None
    for length in lengths:
        for instance in range(instances):
            array = _unsolved_instance(rng, length, env_kind)
            env_cls = ENVIRONMENTS[env_kind][0]
            env = env_cls(
                k=k, max_episode_steps=budget_factor * length * length,
                instruction_costs=costs)
            env.reset(seed=seed)
            env.A, env.tape_env.target = list(array), sorted(array)
            env.v[::2], env.v[1::2] = 0, length - 1
            env.steps_taken = 0
            observation = env._get_obs()
            while True:
                try:
                    facts = parse_observation(render_observation(observation, k), k)
                    action = parse_action(policy(facts), k)
                except (ParseError, Exception) as error:   # noqa: BLE001 - candidate code
                    if first_error is None:
                        first_error = f"{type(error).__name__}: {error}"
                    failures.append((length, "raised"))
                    break
                observation, _r, terminated, truncated, _i = env.step(action)
                if terminated:
                    solved += 1
                    steps.append((length, env.steps_taken))
                    costs_seen.append(env.episode_cost)
                    break
                if truncated:
                    failures.append((length, "budget"))
                    break
    total = len(lengths) * instances
    return {"solved": solved, "total": total,
            "solve_rate": solved / total,
            "mean_steps": float(np.mean([s for _, s in steps])) if steps else float("nan"),
            "mean_cost": float(np.mean(costs_seen)) if costs_seen else float("inf"),
            "instruction_mix": _instruction_mix(policy, k=k, costs=costs, env_kind=env_kind),
            "by_length": {length: sum(1 for solved_length, _ in steps
                                      if solved_length == length)
                          for length in lengths},
            "first_error": first_error,
            "budget_failures": sum(1 for _, why in failures if why == "budget"),
            "raised": sum(1 for _, why in failures if why == "raised")}


def feedback(result):
    """What to tell the model about the current best candidate."""
    if result["solve_rate"] >= 1.0:
        return (f"It sorts every instance, at an average price of "
                f"{result['mean_cost']:.0f} per episode, spending "
                f"{result['instruction_mix']}.")
    if result["first_error"] is None and result["budget_failures"]:
        return (f"It solved {result['solved']}/{result['total']}. "
                f"{result['budget_failures']} runs never finished - the policy is "
                f"looping without making progress. Solved by length: {result['by_length']}.")
    if result["first_error"]:
        return (f"It solved {result['solved']}/{result['total']}. "
                f"The first failure was {result['first_error']}.")
    return f"It solved {result['solved']}/{result['total']}."


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="ornith-1.5:9b")
    parser.add_argument("--rounds", type=int, default=6, help="refinement attempts")
    parser.add_argument("--candidates", type=int, default=3, help="samples per round")
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--lengths", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context", type=int, default=16384,
                        help="context window; the default 4096 is too small for a "
                             "prompt plus a seed program plus a reply")
    parser.add_argument("--seed-program", default=None,
                        help="start from this program instead of from scratch, so two "
                             "cost models can be compared from an identical starting point")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="per-generation timeout; a large model writing code on CPU "
                             "can take many minutes")
    parser.add_argument("--env", choices=sorted(ENVIRONMENTS), default="array",
                        dest="env_kind",
                        help="array has edges to anchor on; ring does not")
    parser.add_argument("--costs", choices=sorted(COST_MODELS), default="uniform",
                        help="instruction price list the candidate is optimising")
    parser.add_argument("--assign-cost", type=float, default=None,
                        help="override the AssignVar price, for a dose-response sweep "
                             "across the MoveVar/AssignVar trade-off")
    parser.add_argument("--out", default=None, help="write the best program here")
    args = parser.parse_args()

    costs = dict(COST_MODELS[args.costs])
    if args.assign_cost is not None:
        costs["AssignVar"] = args.assign_cost
    evaluation = {"lengths": tuple(args.lengths), "instances": args.instances,
                  "costs": costs, "env_kind": args.env_kind}
    api = build_api(costs, args.env_kind)
    print(f"model={args.model} rounds={args.rounds} candidates={args.candidates}")
    print(f"scoring on lengths {args.lengths}, {args.instances} instances each")
    print(f"env {args.env_kind}, cost model {args.costs}: {costs}\n")

    # Reference numbers come from the same instances and budget, so the
    # comparison against a candidate is like for like.
    references = (("insertion", insertion_sort_agent), ("bubble", bubble_sort_agent))
    if args.env_kind == "ring":
        references = (("ring_seam", ring_sort_agent), ("bubble", bubble_sort_agent))
    for name, agent in references:
        scores = evaluate_agent(agent, args.lengths, args.instances, costs=costs,
                                env_kind=args.env_kind)
        print(f"reference {name:10s} solved {scores['solved']}/{scores['total']} "
              f"mean steps {scores['mean_steps']:.1f} mean cost {scores['mean_cost']:.1f}")
    print()

    best, best_source, transcript = None, None, []
    messages = [{"role": "user", "content": api}]

    if args.seed_program:
        with open(args.seed_program) as handle:
            seed_policy, seed_source = extract_policy(handle.read())
        # Keep only the function; the file's header docstring is provenance for
        # readers and would otherwise crowd out the model's room to answer.
        seed_source = seed_source[seed_source.index("def policy"):]
        best = evaluate(seed_policy, **evaluation)
        best_source = seed_source
        print(f"seed program: solved {best['solved']}/{best['total']} "
              f"cost {best['mean_cost']:.1f} mix {best['instruction_mix']}\n")
    started = time.time()
    calls = 0

    for round_index in range(args.rounds):
        for candidate in range(args.candidates):
            try:
                reply = call(args.model, messages, think=args.think, timeout=args.timeout,
                             context=args.context)
                calls += 1
            except Exception as error:                 # noqa: BLE001
                print(f"  backend error: {error}", file=sys.stderr)
                return 1
            try:
                policy, source = extract_policy(reply)
            except ValueError as error:
                print(f"round {round_index} candidate {candidate}: {error}")
                transcript.append((None, str(error)))
                continue
            result = evaluate(policy, **evaluation)
            print(f"round {round_index} candidate {candidate}: "
                  f"solved {result['solved']}/{result['total']} "
                  f"({100*result['solve_rate']:.0f}%) cost {result['mean_cost']:.1f} "
                  f"mix {result['instruction_mix']}"
                  + (f" first_error={result['first_error']}" if result["first_error"] else ""))
            # Correctness first, then cheapness. A cheap policy that does not
            # sort is worthless, so cost only breaks ties among solvers.
            if best is None or (result["solve_rate"], -result["mean_cost"]) > (
                    best["solve_rate"], -best["mean_cost"]):
                best, best_source = result, source
            transcript.append((result, source))


        # Refine from the best program so far rather than the most recent one,
        # so a bad sample does not throw away the round's progress.
        if best is not None:
            messages = [
                {"role": "user", "content": api},
                {"role": "assistant", "content": f"```python\n{best_source}\n```"},
                {"role": "user", "content": feedback(best) +
                 " Rewrite the function so it sorts every instance for a lower total "
                 "price. Make sure every path makes progress towards sorted order so "
                 "the policy cannot loop forever. Reply with the function only."},
            ]

    print(f"\n{calls} generations in {time.time()-started:.0f}s")
    if best:
        print(f"best: solved {best['solved']}/{best['total']} "
              f"({100*best['solve_rate']:.0f}%), mean steps {best['mean_steps']:.1f}, "
              f"mean cost {best['mean_cost']:.1f}, mix {best['instruction_mix']}")
        print(f"\n{best_source}")
        if args.out:
            with open(args.out, "w") as handle:
                handle.write(best_source)
            print(f"\nwritten to {args.out}")
    return 0


INSTRUCTION_NAMES = {0: "SwapWithNext", 1: "MoveVar", 2: "AssignVar"}


def _instruction_mix(policy, k=3, costs=None, n=20, trials=4, env_kind="array"):
    """How the policy spends its instructions - the thing a cost model steers."""
    from collections import Counter
    rng = np.random.default_rng(7)
    counts = Counter()
    for _ in range(trials):
        array = _unsolved_instance(rng, n, env_kind)
        env = ENVIRONMENTS[env_kind][0](k=k, max_episode_steps=20 * n * n,
                                        instruction_costs=costs)
        env.reset(seed=0)
        env.A, env.tape_env.target = list(array), sorted(array)
        env.v[::2], env.v[1::2] = 0, n - 1
        env.steps_taken, env.episode_cost = 0, 0.0
        observation = env._get_obs()
        while True:
            try:
                facts = parse_observation(render_observation(observation, k), k)
                action = parse_action(policy(facts), k)
            except Exception:                          # noqa: BLE001 - candidate code
                return dict(counts)
            counts[INSTRUCTION_NAMES[action[0]]] += 1
            observation, _r, terminated, truncated, _i = env.step(action)
            if terminated or truncated:
                break
    return dict(counts)


def evaluate_agent(agent, lengths, instances, costs=None, env_kind="array"):
    """Score a scripted agent on the same instances a candidate program sees.

    Scripted agents read the observation bits directly rather than the facts
    wrapper, so they need their own loop.
    """
    rng = np.random.default_rng(0)
    solved, steps, episode_costs = 0, [], []
    for length in lengths:
        for _ in range(instances):
            array = _unsolved_instance(rng, length, env_kind)
            env = ENVIRONMENTS[env_kind][0](k=3, max_episode_steps=4 * length * length,
                                            instruction_costs=costs)
            env.reset(seed=0)
            env.A, env.tape_env.target = list(array), sorted(array)
            env.v[::2], env.v[1::2] = 0, length - 1
            env.steps_taken, env.episode_cost = 0, 0.0
            observation = env._get_obs()
            while True:
                observation, _r, terminated, truncated, _i = env.step(agent(observation, 3))
                if terminated:
                    solved += 1
                    steps.append(env.steps_taken)
                    episode_costs.append(env.episode_cost)
                    break
                if truncated:
                    break
    return {"solved": solved, "total": len(lengths) * instances,
            "mean_steps": float(np.mean(steps)) if steps else float("nan"),
            "mean_cost": float(np.mean(episode_costs)) if episode_costs else float("nan")}


if __name__ == "__main__":
    sys.exit(main())
