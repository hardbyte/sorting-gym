"""Stage 0 validation: the rendered text carries the whole interface."""

import numpy as np
import pytest

from sorting_gym.agents import scripted
from sorting_gym.envs.basic_neural_sort_interface import BasicNeuralSortInterfaceEnv
from sorting_gym.text import (
    BITS, SEMANTIC, ParseError, format_action, parse_action, parse_observation,
    render_observation, text_bubble_sort_agent, text_insertion_sort_agent)
from tests.util import _test_sort_agent

STYLES = [SEMANTIC, BITS]


def _random_observations(k, count, seed=0):
    """Observations from random arrays and random pointer placements."""
    rng = np.random.default_rng(seed)
    env = BasicNeuralSortInterfaceEnv(k=k)
    env.reset(seed=int(seed))
    for _ in range(count):
        length = int(rng.integers(1, 12))
        env.A = list(rng.integers(0, 10, length))
        env.v[:] = rng.integers(0, length, k)
        yield env._get_obs()


@pytest.mark.parametrize("style", STYLES)
def test_round_trip_preserves_every_predicate(style):
    """Parsing the rendered text recovers exactly the scripted accessors' view."""
    k = 3
    for obs in _random_observations(k, 300, seed=1):
        facts = parse_observation(render_observation(obs, k, style=style), k, style=style)
        for i in range(k):
            for direction in (-1, +1):
                assert facts.data_neighbour_greater(i, direction) == bool(
                    scripted.data_neighbour_greater(obs, i, direction))
                assert facts.data_neighbour_less(i, direction) == bool(
                    scripted.data_neighbour_less(obs, i, direction))
            for j in range(i + 1, k):
                assert facts.v_less_than(i, j) == bool(scripted.v_less_than(obs, i, j, k))
                assert facts.v_equals(i, j) == bool(scripted.v_equals(obs, i, j, k))
                assert facts.v_greater_than(i, j) == bool(scripted.v_greater_than(obs, i, j, k))
                assert facts.data_less_than(i, j) == bool(scripted.data_less_than(obs, i, j, k))
                assert facts.data_greater_than(i, j) == bool(
                    scripted.data_greater_than(obs, i, j, k))


@pytest.mark.parametrize("style", STYLES)
def test_render_leaks_neither_values_nor_length(style):
    """Arrays that differ in contents and length must render identically.

    The observation is deliberately blind to the values in A, to len(A), and to
    the absolute pointer positions. Rendering must stay blind to them too --
    otherwise the length generalization experiment measures nothing.
    """
    k = 3
    # Same comparison pattern under v = [0, 1, 2], different values and lengths.
    short = [3, 1, 2, 9]
    long = [70, 50, 60, 90] + [99] * 40
    env = BasicNeuralSortInterfaceEnv(k=k)
    env.reset(seed=0)

    env.A, env.v[:] = list(short), [0, 1, 2]
    short_text = render_observation(env._get_obs(), k, style=style)
    env.A, env.v[:] = list(long), [0, 1, 2]
    long_text = render_observation(env._get_obs(), k, style=style)

    assert short_text == long_text


@pytest.mark.parametrize("style", STYLES)
def test_identical_observations_render_identically(style):
    """Rendering is a function of the observation alone."""
    k = 3
    seen = {}
    for obs in _random_observations(k, 400, seed=2):
        key = (obs['neighbour_view_comparisons'].tobytes(),
               obs['pairwise_view_comparisons'].tobytes())
        text = render_observation(obs, k, style=style)
        assert seen.setdefault(key, text) == text


@pytest.mark.parametrize("style", STYLES)
def test_text_bubble_sort_agent(style):
    env = BasicNeuralSortInterfaceEnv(k=3)
    _test_sort_agent(lambda obs, k: text_bubble_sort_agent(obs, k, style=style), env, 1000)


@pytest.mark.parametrize("style", STYLES)
def test_text_insertion_sort_agent(style):
    env = BasicNeuralSortInterfaceEnv(k=3)
    _test_sort_agent(lambda obs, k: text_insertion_sort_agent(obs, k, style=style), env, 1000)


def test_text_agents_match_their_scripted_originals_step_for_step():
    k = 3
    for obs in _random_observations(k, 300, seed=3):
        assert tuple(text_bubble_sort_agent(obs, k)) == tuple(scripted.bubble_sort_agent(obs, k))
        assert tuple(text_insertion_sort_agent(obs, k)) == tuple(
            scripted.insertion_sort_agent(obs, k))


@pytest.mark.parametrize("text,expected", [
    ("SwapWithNext(0)", (0, 0)),
    ("MoveVar(1, +1)", (1, 1, True)),
    ("MoveVar(1, -1)", (1, 1, False)),
    ("  AssignVar(0, 2) ", (2, 0, 2)),
])
def test_parse_action(text, expected):
    assert tuple(parse_action(text, 3)) == expected


@pytest.mark.parametrize("text", [
    "", "Move(0)", "MoveVar(0)", "MoveVar(0, up)", "SwapWithNext(9)",
    "AssignVar(0, -1)", "SwapWithNext(0", "let me think... MoveVar(0, +1)",
])
def test_parse_action_rejects_malformed_output(text):
    with pytest.raises(ParseError):
        parse_action(text, 3)


@pytest.mark.parametrize("action", [(0, 2), (1, 0, True), (1, 0, False), (2, 1, 2)])
def test_format_action_round_trips(action):
    assert tuple(parse_action(format_action(action), 3)) == action


def test_parse_observation_rejects_malformed_text():
    with pytest.raises(ParseError):
        parse_observation("v0: something odd | no right neighbour", 3)


def test_demonstrations_are_not_all_the_same_action():
    """Few-shot demonstrations must show contrast.

    Sampling random pointer placements and taking the first few hits returns
    the same instruction every time, because one action covers ~61% of the
    expert's choices there. A prompt full of identical answers teaches a
    constant policy - the exact failure it then looks like the model invented.
    """
    from sorting_gym.text.prompts import sample_demonstrations
    demonstrations = sample_demonstrations(3, 5, seed=1)
    assert len(demonstrations) == 5
    assert len({action for _text, action in demonstrations}) >= 4


def test_demonstration_actions_match_the_rendered_state():
    """Each demonstrated action is what the expert really does in that state."""
    from sorting_gym.text.prompts import sample_demonstrations
    from sorting_gym.text.agents import insertion_sort_policy
    for text, action in sample_demonstrations(3, 8, seed=2):
        assert insertion_sort_policy(parse_observation(text, 3)) == action
