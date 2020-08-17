
import numpy as np
from gym import spaces, ActionWrapper
from gym.spaces import flatten_space, flatdim, unflatten, flatten

from sorting_gym import DiscreteParametric


class SimpleActionSpace(ActionWrapper):
    """Expose a flat Box action space instead of a parametric action space.

    Example::

        >>> isinstance(SimpleActionSpace(env).action_space, Box)
        True

    Note that sampling from a Box is not the same as flattening samples from a richer
    subspace. To draw action space samples from a `SimpleActionSpace` call
    `SimpleActionSpace.action_space_sample()`

    """
    def __init__(self, env):
        assert isinstance(env.action_space, DiscreteParametric), ("expected DiscreteParametric action space, got {}".format(type(env.action_space)))
        super(SimpleActionSpace, self).__init__(env)
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
