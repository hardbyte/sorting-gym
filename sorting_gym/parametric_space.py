from gym import Space
from gym.spaces import Discrete


class DiscreteParametric(Space):
    """
    A disjoint set of spaces conditioned by a discrete space.

    Example usage:

    >>> action_space = DiscreteParametric(2, ([Discrete(2), Discrete(3)]))

    """
    def __init__(self, n, spaces):
        self.parameter_space = Discrete(n)
        self.disjoint_spaces = spaces
        for space in spaces:
            assert isinstance(space, Space), "Elements of the DiscreteParametric must be instances of gym.Space"
        super().__init__(None, None)

    def seed(self, seed=None):
        self.parameter_space.seed(seed)
        [space.seed(seed) for space in self.disjoint_spaces]

    def sample(self):
        parameter_sample = self.parameter_space.sample()
        return tuple([parameter_sample, self.disjoint_spaces[parameter_sample].sample()])

    def contains(self, x):
        if isinstance(x, list):
            x = tuple(x)  # Promote list to tuple for contains check
        parameter, args = x
        return self.parameter_space.contains(parameter) and \
               self.disjoint_spaces[parameter].contains(args)

    def __repr__(self):
        return f"DiscreteParametric({self.parameter_space.n}, [" +\
               ", ".join([str(s) for s in self.disjoint_spaces]) + "])"

    def to_jsonable(self, sample_n):
        # serialize as list-repr
        res = []
        for parameter, subspace in sample_n:
            res.append([parameter, self.disjoint_spaces[parameter].to_jsonable(subspace)])
        return res

    def from_jsonable(self, sample_n):
        raise NotImplemented("Got bored")
        #return [sample for sample in zip(*[space.from_jsonable(sample_n[i]) for i, space in enumerate(self.spaces)])]

    def __getitem__(self, index):
        return self.disjoint_spaces[index]

    def __len__(self):
        return self.parameter_space.n

    def __eq__(self, other):
        return isinstance(other, DiscreteParametric) and self.disjoint_spaces == other.disjoint_spaces
