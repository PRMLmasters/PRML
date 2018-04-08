import numpy as np

from ..utils.function import comb
from .bernoulli import BernoulliDistribution


class BinomialDistribution(BernoulliDistribution):

    def __init__(self, mu, n):
        super(BernoulliDistribution, self).__init__(mu)
        self.n = n

    def trial(self):
        return np.random.choice(list(self.prob.keys()), size=self.n,
                                p=list(self.prob.values()))

    @property
    def expected_value(self):
        return self.n * prob[1]

    @property
    def variance(self):
        return self.n * self.prob[1] * self.prob[0]

    def probability(self, m):
        """The probability of getting m times of 1 in n times.
           This is known as Bin(m|n, mu).

        """

        return comb(self.n, m) * (self.mu ** m) * ((1 - self.mu) ** (self.n - m))
