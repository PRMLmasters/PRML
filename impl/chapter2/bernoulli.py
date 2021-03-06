import numpy as np

from .distribution import DiscreteDistribution


class BernoulliDistribution(DiscreteDistribution):

    def __init__(self, prob):
        super(BernoulliDistribution, self).__init__(prob)
        self.prob = {
            0: 1 - prob,
            1: prob,
        }

    def trial(self):
        return np.random.choice(list(self.prob.keys()),
                                p=list(self.prob.values()))

    @property
    def expected_value(self):
        return sum([value * prob for value, prob in self.prob.items()])

    @property
    def variance(self):
        e_x_2 = sum([value**2 * prob for value, prob in self.prob.items()])
        return e_x_2 - self.expected_value**2
