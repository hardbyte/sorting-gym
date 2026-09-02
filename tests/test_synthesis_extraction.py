"""Pulling the function out of a model's reply.

Models narrate around their code and a long reply can be cut off before its
closing fence. Half the candidates in one run were lost to extraction rather
than to writing a bad policy, so the shapes are pinned here.
"""

import importlib.util
import pathlib

import pytest

SYNTHESIS = pathlib.Path(__file__).parent.parent / "examples" / "llm_synthesis.py"
GOOD = 'def policy(facts):\n    return "MoveVar(0, +1)"'


def _module():
    spec = importlib.util.spec_from_file_location("llm_synthesis", SYNTHESIS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("reply", [
    f"Here you go:\n```python\n{GOOD}\n```\nDone.",
    f"Reasoning first, then cut off mid-reply.\n```python\n{GOOD}",
    f"Let me think. Step one. Step two.\n{GOOD}",
    f"Let's refine this.\n```python\n{GOOD}\n```",
    f"```python\nx = 1\n```\nand the policy:\n```python\n{GOOD}\n```",
])
def test_extracts_the_function(reply):
    policy, _source = _module().extract_policy(reply)
    assert policy(None) == "MoveVar(0, +1)"


@pytest.mark.parametrize("reply", [
    "I cannot help with that.",
    "```python\nx = 1\n```",
    "",
])
def test_rejects_a_reply_with_no_policy(reply):
    with pytest.raises(ValueError):
        _module().extract_policy(reply)


def test_rejects_a_policy_that_does_not_compile():
    with pytest.raises(ValueError, match="no compilable"):
        _module().extract_policy("def policy(facts):\n    return 'unterminated")
