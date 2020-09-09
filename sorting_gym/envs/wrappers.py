from typing import Union, List

import numpy as np
from gym import spaces, ActionWrapper
from gym.spaces import flatten_space, flatdim, unflatten, flatten

from sorting_gym import DiscreteParametric


def merge_discrete_spaces(input_spaces: List[Union[spaces.Discrete, spaces.Tuple, spaces.MultiBinary]]) -> spaces.MultiDiscrete:
    """
    Merge nested Discrete, and MultiBinary spaces into a single MultiDiscrete space

    TODO could also add support for MultiDiscrete

    :param input_spaces:
    :return:
    """
    return spaces.MultiDiscrete(_discrete_dims(input_spaces))


def _discrete_dims(input_spaces: Union[spaces.Discrete, spaces.Tuple, spaces.MultiBinary]):
    sizes = []
    for space in input_spaces:
        if isinstance(space, spaces.Discrete):
            sizes.append(space.n)
        elif isinstance(space, spaces.MultiBinary):
            sizes.extend([2 for _ in range(space.n)])
        elif isinstance(space, spaces.MultiDiscrete):
            sizes.extend(space.nvec)
        elif isinstance(space, spaces.Tuple):
            sizes.extend(_discrete_dims(space.spaces))
    return sizes


def _discrete_unflatten(argument_space, args):
    """

    :param argument_space:
    :param args:
    :return:
    """
    res = []
    args = list(args)
    while len(args) > 0:
        if isinstance(argument_space, spaces.Discrete):
            res.append(args.pop(0))
        elif isinstance(argument_space, spaces.MultiDiscrete):
            res.append(args[:argument_space.shape[0]])
            del args[:argument_space.shape[0]]
        elif isinstance(argument_space, spaces.MultiBinary):
            res.append(args[:argument_space.n])
            del args[:argument_space.shape[0]]
        elif isinstance(argument_space, spaces.Tuple):
            _num_tuple_args = _discrete_dims(argument_space.spaces)
            res.append(args[:len(_num_tuple_args)])
            del args[:len(_num_tuple_args)]
        else:
            raise NotImplemented

    return res


class DisjointMultiDiscreteActionSpaceWrapper(ActionWrapper):
    """Expose a MultiDiscrete action space for each disjoint action space instead of a more complex nested space.

    Wrapping a discrete parametric space with the following disjoint spaces:

        Discrete(k),
        Tuple([Discrete(k), MultiBinary(1)]),
        Tuple([Discrete(k), Discrete(k)]),

    should result in output spaces of:

        MultiDiscrete([k]),
        MultiDiscrete([k, 2]),
        MultiDiscrete([k, k]

    """

    def __init__(self, env):
        assert isinstance(env.action_space, DiscreteParametric), (
            "expected DiscreteParametric action space, got {}".format(type(env.action_space)))
        super(DisjointMultiDiscreteActionSpaceWrapper, self).__init__(env)
        self.parametric_space: DiscreteParametric = env.action_space

        # Construct the modified disjoint spaces
        self.disjoint_action_spaces = [merge_discrete_spaces([s]) for s in self.parametric_space.disjoint_spaces]
        self.action_space = DiscreteParametric(env.action_space.parameter_space.n, self.disjoint_action_spaces)

    def action(self, action):
        """
        Convert an action using the merged MultiDiscrete disjoint space into a DiscreteParametric action.

        """
        assert self.action_space.contains(action), "Given action is not valid in this action space"
        # Get the discrete parameter value
        parameter = action[0]
        # The args should be a valid MultiDiscrete sample for the given parameter. Note
        # MultiDiscrete samples are ndarrays of dtype np.int64.
        args = action[1:]
        assert self.disjoint_action_spaces[parameter].contains(np.array(args, dtype=np.int64))

        # TODO the args need to be converted back into their original nested form
        #
        output_space = self.env.action_space.disjoint_spaces[parameter]

        raise NotImplemented

        assert self.env.action_space.contains(transformed_action)
        return tuple(transformed_action)


class MultiDiscreteActionSpaceWrapper(ActionWrapper):
    """Expose a single MultiDiscrete action space instead of a DiscreteParametric action space.


    """
    def __init__(self, env):
        assert isinstance(env.action_space, DiscreteParametric), ("expected DiscreteParametric action space, got {}".format(type(env.action_space)))
        super(MultiDiscreteActionSpaceWrapper, self).__init__(env)
        parametric_space: DiscreteParametric = env.action_space

        # Construct a space from the parametric space's parameter_space and disjoint spaces
        self.action_space = merge_discrete_spaces([parametric_space.parameter_space] + list(parametric_space.disjoint_spaces))

    def action(self, action):
        """Convert a MultiDiscrete action into a DiscreteParametric action."""
        # Get the discrete parameter value

        parameter = np.argmax(action[0])
        argument_space = self.env.action_space[parameter]

        # Convert the appropriate args for the disjoint space using the parameter
        start_index = 1 + len(_discrete_dims(self.env.action_space.disjoint_spaces[:parameter]))
        end_index = 1 + len(_discrete_dims(self.env.action_space.disjoint_spaces[:parameter + 1]))

        # Our discrete arguments for the disjoint space
        args = action[start_index:end_index]

        disjoint_args = _discrete_unflatten(argument_space, args)

        # Make the final flat tuple
        transformed_action = [parameter]
        if isinstance(disjoint_args, (tuple, list)):
            transformed_action.extend(disjoint_args)
        else:
            transformed_action.append(disjoint_args)

        assert self.env.action_space.contains(transformed_action)
        return tuple(transformed_action)


class BoxActionSpaceWrapper(ActionWrapper):
    """Expose a flat Box action space instead of a parametric action space.

    Example::

        >>> isinstance(BoxActionSpaceWrapper(env).action_space, Box)
        True

    Note that sampling from a Box is not the same as flattening samples from a richer
    subspace. To draw action space samples from a `SimpleActionSpace` call
    `SimpleActionSpace.action_space_sample()`

    """
    def __init__(self, env):
        assert isinstance(env.action_space, DiscreteParametric), ("expected DiscreteParametric action space, got {}".format(type(env.action_space)))
        super(BoxActionSpaceWrapper, self).__init__(env)
        parametric_space: DiscreteParametric = env.action_space

        # Construct a space from the parametric space's parameter_space and disjoint spaces
        self.action_space = flatten_space(spaces.Tuple([parametric_space.parameter_space] +
                                                       list(parametric_space.disjoint_spaces)))

        self.disjoint_sizes = [flatdim(space) for space in parametric_space.disjoint_spaces]

    def action(self, action):
        """Convert a flattened action into a parametric space."""
        # Get the discrete parameter value
        num_disjoint_spaces = len(self.env.action_space)
        parameter = np.argmax(action[:num_disjoint_spaces])
        argument_space = self.env.action_space[parameter]

        # Now we need to index the appropriate args for the disjoint space using the parameter
        start_index = num_disjoint_spaces
        start_index += sum(self.disjoint_sizes[:parameter])
        end_index = start_index + self.disjoint_sizes[parameter]

        # Flattened arguments for the disjoint space
        args = action[start_index:end_index]

        try:
            disjoint_args = unflatten(argument_space, args)
        except IndexError as e:
            # Very likely the args are invalid for the wrapped space e.g. a Discrete(2) getting all zeros.
            msg = "Failed to unflatten arguments to wrapped space of " + str(argument_space)
            raise ValueError(msg) from e

        # Make the final flat tuple
        transformed_action = [parameter]
        if isinstance(disjoint_args, tuple):
            transformed_action.extend(disjoint_args)
        else:
            transformed_action.append(disjoint_args)

        assert self.env.action_space.contains(transformed_action)
        return tuple(transformed_action)

    def reverse_action(self, action):
        """Convert a wrapped action (e.g. from a DiscreteParametric) into a flattened action"""
        parameter = action[0]
        result = np.zeros(self.action_space.shape[0], dtype=self.action_space.dtype)
        result[parameter] = 1.0
        start_index = len(self.env.action_space)
        start_index += sum(self.disjoint_sizes[:parameter])
        end_index = start_index + self.disjoint_sizes[parameter]
        result[start_index:end_index] = flatten(self.env.action_space[parameter], action[1:])
        assert self.action_space.contains(result)
        return result

    def action_space_sample(self):
        rich_sample = self.env.action_space.sample()
        assert self.env.action_space.contains(rich_sample)
        return self.reverse_action(rich_sample)
