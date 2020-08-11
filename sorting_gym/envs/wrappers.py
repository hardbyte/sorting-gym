
import numpy as np
from gym import spaces, ActionWrapper
from gym.spaces import flatten_space, flatdim, unflatten

from sorting_gym import DiscreteParametric


class SimpleActionSpace(ActionWrapper):
    """Expose a flat Box action space instead of a parametric action space.

    Example::

        >>> isinstance(SimpleActionSpace(env).action_space, Box)
        True

    """
    def __init__(self, env):
        assert isinstance(env.action_space, DiscreteParametric), ("expected DiscreteParametric action space, got {}".format(type(env.action_space)))
        super(SimpleActionSpace, self).__init__(env)
        parametric_space: DiscreteParametric = env.action_space

        # Construct a space from the parametric space's parameter_space and disjoint spaces
        self.action_space = flatten_space(spaces.Tuple([parametric_space.parameter_space] +
                                                       list(parametric_space.disjoint_spaces)))

        self.disjoint_sizes = [flatdim(space) for space in parametric_space.disjoint_spaces]

        print(self.action_space)

    def action(self, action):
        # Get the discrete parameter value
        num_disjoint_spaces = self.env.action_space.parameter_space.n
        parameter = np.argmax(action[:num_disjoint_spaces])
        argument_space = self.env.action_space.disjoint_spaces[parameter]

        try:
            argument_sizes = [flatdim(s) for s in argument_space]
        except TypeError:
            argument_sizes = [flatdim(argument_space)]

        # Now we need to index the appropriate args for the disjoint space using the parameter
        start_index = end_index = num_disjoint_spaces
        if parameter > 0:
            start_index += sum(self.disjoint_sizes[:parameter-1])
            end_index += sum(self.disjoint_sizes[:parameter])

        # Flattened arguments for the disjoint space
        args = []
        for argument_size in argument_sizes:
            # Get the ith argument for this space
            args.append(action[start_index: start_index + argument_size])
            start_index += argument_size

        # unflatten the args and concat to finish transforming the action
        if len(argument_sizes) == 1:
            args = args[0]
        disjoint_args = unflatten(argument_space, args)

        transformed_action = tuple([parameter] + [disjoint_args])
        assert self.env.action_space.contains(transformed_action)
        return transformed_action

    def reverse_action(self, action):
        pass