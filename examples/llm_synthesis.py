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

from sorting_gym.agents.scripted import bubble_sort_agent, insertion_sort_agent
from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv
from sorting_gym.text import ParseError, parse_action, parse_observation, render_observation

OLLAMA_URL = "http://localhost:11434/api/chat"
CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)

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
  facts.at_left_edge(i)        -> bool   v_i is at the first element
  facts.at_right_edge(i)       -> bool   v_i is at the last element

Return exactly one instruction as a string:

  "SwapWithNext(i)"   swap A[v_i] with A[v_i + 1]
  "MoveVar(i, +1)"    move v_i one place right (stops at the end)
  "MoveVar(i, -1)"    move v_i one place left (stops at the start)
  "AssignVar(i, j)"   set v_i = v_j

At the start of an episode v0 and v2 are at the first element, v1 at the last.
The episode ends as soon as A is sorted, and fewer instructions is better. Your
function is called fresh each step and keeps no state between calls, so every
decision must come from `facts` alone.

Write one function and nothing else:

```python
def policy(facts):
    ...
    return "MoveVar(0, +1)"
```'''


def call(model, messages, timeout=900, think=False, max_tokens=4000):
    payload = json.dumps({
        "model": model, "messages": messages, "stream": False, "think": think,
        "options": {"temperature": 0.7, "num_predict": max_tokens},
    }).encode()
    request = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["message"]["content"]


def extract_policy(reply):
    """Compile the model's reply into a callable, or raise ValueError."""
    blocks = CODE_BLOCK.findall(reply)
    source = blocks[0] if blocks else reply
    if "def policy" not in source:
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


def evaluate(policy, k=3, lengths=(5, 10, 20), instances=20, seed=0, budget_factor=4):
    """Solve rate and step counts on real instances, plus the first error seen."""
    rng = np.random.default_rng(seed)
    solved, steps, failures, first_error = 0, [], [], None
    for length in lengths:
        for instance in range(instances):
            array = list(rng.integers(0, 10, length))
            while array == sorted(array):
                array = list(rng.integers(0, 10, length))
            env = BasicNeuralSortInterfaceEnv(
                k=k, max_episode_steps=budget_factor * length * length)
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
                    break
                if truncated:
                    failures.append((length, "budget"))
                    break
    total = len(lengths) * instances
    return {"solved": solved, "total": total,
            "solve_rate": solved / total,
            "mean_steps": float(np.mean([s for _, s in steps])) if steps else float("nan"),
            "by_length": {length: sum(1 for l, _ in steps if l == length) for length in lengths},
            "first_error": first_error,
            "budget_failures": sum(1 for _, why in failures if why == "budget"),
            "raised": sum(1 for _, why in failures if why == "raised")}


def feedback(result):
    """What to tell the model about a candidate that did not work."""
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
    parser.add_argument("--out", default=None, help="write the best program here")
    args = parser.parse_args()

    evaluation = {"lengths": tuple(args.lengths), "instances": args.instances}
    print(f"model={args.model} rounds={args.rounds} candidates={args.candidates}")
    print(f"scoring on lengths {args.lengths}, {args.instances} instances each\n")

    # Reference numbers come from the same instances and budget, so the
    # comparison against a candidate is like for like.
    for name, agent in (("insertion", insertion_sort_agent), ("bubble", bubble_sort_agent)):
        scores = evaluate_agent(agent, **evaluation)
        print(f"reference {name:10s} solved {scores['solved']}/{scores['total']} "
              f"mean steps {scores['mean_steps']:.1f}")
    print()

    best, best_source, transcript = None, None, []
    messages = [{"role": "user", "content": API}]
    started = time.time()
    calls = 0

    for round_index in range(args.rounds):
        for candidate in range(args.candidates):
            try:
                reply = call(args.model, messages, think=args.think)
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
                  f"({100*result['solve_rate']:.0f}%) by length {result['by_length']}"
                  + (f" first_error={result['first_error']}" if result["first_error"] else ""))
            if best is None or result["solve_rate"] > best["solve_rate"]:
                best, best_source = result, source
            transcript.append((result, source))

        if best and best["solve_rate"] >= 1.0:
            print("\nsolved every instance; stopping early")
            break
        # Refine from the best program so far rather than the most recent one,
        # so a bad sample does not throw away the round's progress.
        if best is not None:
            messages = [
                {"role": "user", "content": API},
                {"role": "assistant", "content": f"```python\n{best_source}\n```"},
                {"role": "user", "content": feedback(best) +
                 " Rewrite the function so it sorts every instance. Make sure every "
                 "path makes progress towards sorted order so the policy cannot loop "
                 "forever. Reply with the function only."},
            ]

    print(f"\n{calls} generations in {time.time()-started:.0f}s")
    if best:
        print(f"best: solved {best['solved']}/{best['total']} "
              f"({100*best['solve_rate']:.0f}%), mean steps {best['mean_steps']:.1f}")
        print(f"\n{best_source}")
        if args.out:
            with open(args.out, "w") as handle:
                handle.write(best_source)
            print(f"\nwritten to {args.out}")
    return 0


def evaluate_agent(agent, lengths, instances):
    """Score a scripted agent on the same instances a candidate program sees.

    Scripted agents read the observation bits directly rather than the facts
    wrapper, so they need their own loop.
    """
    rng = np.random.default_rng(0)
    solved, steps = 0, []
    for length in lengths:
        for _ in range(instances):
            array = list(rng.integers(0, 10, length))
            while array == sorted(array):
                array = list(rng.integers(0, 10, length))
            env = BasicNeuralSortInterfaceEnv(k=3, max_episode_steps=4 * length * length)
            env.reset(seed=0)
            env.A, env.tape_env.target = list(array), sorted(array)
            env.v[::2], env.v[1::2] = 0, length - 1
            env.steps_taken = 0
            observation = env._get_obs()
            while True:
                observation, _r, terminated, truncated, _i = env.step(agent(observation, 3))
                if terminated:
                    solved += 1
                    steps.append(env.steps_taken)
                    break
                if truncated:
                    break
    return {"solved": solved, "total": len(lengths) * instances,
            "mean_steps": float(np.mean(steps)) if steps else float("nan")}


if __name__ == "__main__":
    sys.exit(main())
