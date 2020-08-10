

from gym import spaces, ActionWrapper
from gym.spaces import flatten_space

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
                                                       list(parametric_space.dijoint_spaces)))
        print(self.action_space)

    def action(self, action):
        raise NotImplemented("TODO")

        assert np.all(np.greater_equal(action, self.a)), (action, self.a)
        assert np.all(np.less_equal(action, self.b)), (action, self.b)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low)*((action - self.a)/(self.b - self.a))
        action = np.clip(action, low, high)
        return action
