"""Frozen-LLM baseline on the neural sorting interface: a go/no-go probe.

Two modes, cheapest first.

  probe     One generation per state. Measures whether the model can read the
            rendered observation at all: how often its output parses, and how
            often it matches a reference policy that is known to solve the task
            through this same text channel.

  episodes  Full rollouts. Only worth running for configurations that survive
            the probe, because a failing episode burns the entire step budget.

The point is to find out whether RL has anything to bootstrap from. If a frozen
model never solves the smallest instances, every GRPO group scores identically
and training starts from zero signal.

Criteria, fixed before running anything, because a threshold chosen after the
fact is a threshold chosen to be passed:

  Probe
    strict parse >= 90%          constrained decoding is not needed
    agreement well above the     the model is reading the observation rather
    majority-action baseline     than guessing one instruction every turn
    (~38% on expert states)

  Episodes, only if the probe passes
    >= 30% of n=4 solved         GO: RL has something to bootstrap from
    some but < 30% solved        MARGINAL: needs an SFT cold start first
    0 solved at n=4              NO-GO: every GRPO group scores identically

Agreement is a proxy, not ground truth: a model could disagree with the
reference and still sort by some other valid route. High agreement is therefore
strong evidence to continue, while low agreement on its own is not evidence to
stop -- only zero episode solves is.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

import numpy as np

from sorting_gym.agents.scripted import bubble_sort_agent, insertion_sort_agent
from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv
from sorting_gym.text import (
    BITS, SEMANTIC, ParseError, action_from_tool_call, parse_action, render_observation,
    tool_schemas)
from sorting_gym.text.prompts import build_messages, sample_demonstrations

OLLAMA_URL = "http://localhost:11434/api/chat"
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
# Chat models like to dress a bare answer in markdown. That is presentation,
# not a malformed instruction, so it is normalised away before strict parsing.
MARKDOWN = re.compile(r"[`*_]+")
ANY_INSTRUCTION = re.compile(r"(SwapWithNext|MoveVar|AssignVar)\s*\([^)]*\)")


class BackendError(RuntimeError):
    """The model could not be reached. Never report this as a score."""


class Backend:
    """Chat completion against a local ollama server."""

    def __init__(self, model, temperature=0.0, timeout=600, tools=None):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.tools = tools
        self.calls = 0

    def __call__(self, messages, max_tokens=200):
        request_body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": self.temperature, "num_predict": max_tokens},
        }
        if self.tools:
            request_body["tools"] = self.tools
        payload = json.dumps(request_body).encode()
        request = urllib.request.Request(
            OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace").strip()
            raise BackendError(
                f"ollama returned {error.code} for model {self.model!r}: {detail}") from error
        except (urllib.error.URLError, OSError) as error:
            raise BackendError(f"could not reach ollama at {OLLAMA_URL}: {error}") from error
        self.calls += 1
        message = body["message"]
        if self.tools:
            # Returned as a call so the caller can tell a constrained decode
            # apart from the model answering in prose anyway.
            calls = message.get("tool_calls") or []
            if calls:
                function = calls[0]["function"]
                return {"name": function["name"], "arguments": function.get("arguments", {})}
        return message["content"]


def _token_budget(backend, chain_of_thought):
    if backend.tools:
        return 200
    return 300 if chain_of_thought else 30


def interpret(raw, k):
    """Also accepts a tool call dict, in which case the schema guarantees form."""
    if isinstance(raw, dict):
        try:
            action = action_from_tool_call(raw["name"], raw["arguments"], k)
        except ParseError:
            return None, None
        return action, action

    text = MARKDOWN.sub("", THINK_BLOCK.sub("", raw)).strip().rstrip(".")
    strict = None
    try:
        strict = parse_action(text, k)
    except ParseError:
        pass

    lenient = strict
    if lenient is None:
        for match in reversed(list(ANY_INSTRUCTION.finditer(text))):
            try:
                lenient = parse_action(match.group(0), k)
                break
            except ParseError:
                continue
    return strict, lenient


def _random_states(k, count, seed, base=10, lengths=(4, 9)):
    """Random arrays with randomly placed pointers.

    Cheap and diverse, but off-distribution: many of these states are not
    reachable from a reset by any sensible policy.
    """
    rng = np.random.default_rng(seed)
    env = BasicNeuralSortInterfaceEnv(k=k, base=base)
    env.reset(seed=seed)
    for _ in range(count):
        length = int(rng.integers(*lengths))
        env.A = list(rng.integers(0, base, length))
        env.v[:] = rng.integers(0, length, k)
        yield env._get_obs()


def _expert_states(k, count, seed, base=10, lengths=(4, 9)):
    """States actually visited along expert trajectories.

    This is the distribution a policy meets in a real episode, so agreement
    measured here is the number that predicts episode performance.
    """
    rng = np.random.default_rng(seed)
    collected = []
    while len(collected) < count:
        length = int(rng.integers(*lengths))
        array = list(rng.integers(0, base, length))
        if array == sorted(array):
            continue
        env = BasicNeuralSortInterfaceEnv(k=k, base=base)
        env.reset(seed=seed)
        env.A, env.tape_env.target = array, sorted(array)
        env.v[::2], env.v[1::2] = 0, length - 1
        env.steps_taken = 0
        observation = env._get_obs()
        while len(collected) < count:
            collected.append(observation)
            observation, _r, terminated, truncated, _i = env.step(
                insertion_sort_agent(observation, k))
            if terminated or truncated:
                break
    rng.shuffle(collected)
    return collected[:count]


def run_probe(backend, k, style, shots, chain_of_thought, states, seed, verbose):
    demonstrations = sample_demonstrations(k, shots, style=style, seed=seed + 1) if shots else ()
    counts = Counter()
    for observation in states:
        text = render_observation(observation, k, style=style)
        messages = build_messages(text, k, style=style, demonstrations=demonstrations,
                                  chain_of_thought=chain_of_thought,
                                  tool_calling=bool(backend.tools))
        raw = backend(messages, max_tokens=_token_budget(backend, chain_of_thought))
        strict, lenient = interpret(raw, k)
        counts["total"] += 1
        counts["strict"] += strict is not None
        counts["lenient"] += lenient is not None
        if lenient is not None:
            counts["insertion"] += tuple(lenient) == tuple(insertion_sort_agent(observation, k))
            counts["bubble"] += tuple(lenient) == tuple(bubble_sort_agent(observation, k))
        if verbose:
            shown = raw if isinstance(raw, dict) else raw.strip()[:40]
            print(f"    {text.splitlines()[0][:40]:42s} -> {shown!r}")
    return counts


def run_episodes(backend, k, style, shots, chain_of_thought, n, instances, seed, verbose,
                 max_steps=None):
    demonstrations = sample_demonstrations(k, shots, style=style, seed=seed + 1) if shots else ()
    rng = np.random.default_rng(seed)
    results = []
    for instance in range(instances):
        env = BasicNeuralSortInterfaceEnv(k=k, max_episode_steps=max_steps)
        env.reset(seed=seed + instance)
        array = list(rng.integers(0, 10, n))
        while array == sorted(array):
            array = list(rng.integers(0, 10, n))
        env.A, env.tape_env.target = array, sorted(array)
        env.v[::2], env.v[1::2] = 0, n - 1
        env.steps_taken = 0

        observation, malformed, terminated, truncated = env._get_obs(), 0, False, False
        while not (terminated or truncated):
            text = render_observation(observation, k, style=style)
            messages = build_messages(text, k, style=style, demonstrations=demonstrations,
                                      chain_of_thought=chain_of_thought,
                                      tool_calling=bool(backend.tools))
            raw = backend(messages, max_tokens=_token_budget(backend, chain_of_thought))
            _strict, action = interpret(raw, k)
            if action is None:
                # A malformed reply still costs a step, mirroring the format
                # penalty an RL run would apply.
                malformed += 1
                action = (1, 0, True)
            observation, reward, terminated, truncated, _info = env.step(action)

        results.append({"solved": bool(terminated), "steps": env.steps_taken,
                        "malformed": malformed})
        if verbose:
            status = "solved" if terminated else "FAILED"
            print(f"    instance {instance}: {status} in {env.steps_taken} steps "
                  f"({malformed} malformed)")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--mode", choices=("probe", "episodes"), default="probe")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--states", type=int, default=40, help="probe: states per config")
    parser.add_argument("--instances", type=int, default=5, help="episodes: instances per config")
    parser.add_argument("--n", type=int, nargs="+", default=[4, 6],
                        help="episodes: array lengths")
    parser.add_argument("--styles", nargs="+", default=[SEMANTIC, BITS])
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 5])
    parser.add_argument("--cot", type=int, nargs="+", default=[0],
                        help="0 for direct answers, 1 for chain of thought")
    parser.add_argument("--states-from", choices=("expert", "random"), default="expert",
                        help="expert: states visited on expert trajectories (on-distribution)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="episode step budget (default: the env's 4n^2)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tools", action="store_true",
                        help="use native tool calling instead of free text")
    parser.add_argument("--timeout", type=int, default=600,
                        help="per-generation timeout; CPU-only inference is slow")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.tools and any(shots > 0 for shots in args.shots):
        # Demonstrations are plain assistant text; mixing them with tool mode
        # would show the model a format it is not being asked to produce.
        print("error: --tools does not yet support few-shot demonstrations, "
              "which would need to be tool calls themselves. Use --shots 0.",
              file=sys.stderr)
        return 1

    backend = Backend(args.model, timeout=args.timeout,
                      tools=tool_schemas(args.k) if args.tools else None)
    started = time.time()
    # Fail fast and loudly: a table of zeroes from an unreachable model reads
    # exactly like a model that cannot do the task.
    try:
        # Health check without tools: an unreachable model must not look like a
        # model that cannot use them.
        Backend(args.model)([{"role": "user", "content": "reply with OK"}], max_tokens=5)
    except BackendError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"model={args.model} mode={args.mode} k={args.k}\n")

    if args.mode == "probe":
        print(f"{'style':10s} {'shots':>5s} {'cot':>3s} {'strict%':>8s} {'parsed%':>8s} "
              f"{'=insert%':>9s} {'=bubble%':>9s}")
        for style in args.styles:
            for shots in args.shots:
                for cot in args.cot:
                    sampler = (_expert_states if args.states_from == "expert"
                               else _random_states)
                    states = list(sampler(args.k, args.states, args.seed))
                    counts = run_probe(backend, args.k, style, shots, bool(cot), states,
                                       args.seed, args.verbose)
                    total = max(counts["total"], 1)
                    print(f"{style:10s} {shots:5d} {cot:3d} "
                          f"{100*counts['strict']/total:7.1f}% {100*counts['lenient']/total:7.1f}% "
                          f"{100*counts['insertion']/total:8.1f}% {100*counts['bubble']/total:8.1f}%")
    else:
        print(f"{'style':10s} {'shots':>5s} {'cot':>3s} {'n':>3s} {'solved':>8s} "
              f"{'mean steps':>11s} {'malformed':>10s}")
        for style in args.styles:
            for shots in args.shots:
                for cot in args.cot:
                    for n in args.n:
                        results = run_episodes(backend, args.k, style, shots, bool(cot), n,
                                               args.instances, args.seed, args.verbose,
                                               max_steps=args.max_steps)
                        solved = [r for r in results if r["solved"]]
                        steps = np.mean([r["steps"] for r in solved]) if solved else float("nan")
                        malformed = np.mean([r["malformed"] for r in results]) if results else 0
                        print(f"{style:10s} {shots:5d} {cot:3d} {n:3d} "
                              f"{len(solved):3d}/{len(results):<4d} {steps:11.1f} {malformed:10.1f}")

    print(f"\n{backend.calls} generations in {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BackendError as error:
        print(f"\nerror: {error}", file=sys.stderr)
        sys.exit(1)
