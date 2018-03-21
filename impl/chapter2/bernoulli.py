import numpy as np

from .distribution import Distribution


class BernoulliDist(Distribution):

    def __init__(self, prob):
        if 0 < prob < 1:
            self.prob = prob
        else:
            print('0 < probability < 1')

    def trial(self):
        weight = [self.prob, 1 - self.prob]
        return np.random.choice([1, 0], p=weight)

    def expected_value(self):
        pass

    def variance(self):
        pass
