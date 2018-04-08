import numpy as np

from ..utils.function import comb
from .bernoulli import BernoulliDistribution


class BinomialDistribution(BernoulliDistribution):

    def __init__(self, mu, n_trials):
        super(BernoulliDistribution, self).__init__(mu)
        self.n_trials = n_trials

    def trial(self):
        return np.random.choice(list(self.prob.keys()), size=self.n_trials,
                                p=list(self.prob.values()))

    @property
    def expected_value(self):
        return self.n_trials * prob[1]

    @property
    def variance(self):
        return self.n_trials * self.prob[1] * self.prob[0]

    def probability(self, m):
        """The probability of getting m times of 1 in n times.
           This is known as Bin(m|n, mu).

        """

        return comb(self.n_trials, m) * (self.mu ** m) * ((1 - self.mu) ** (self.n_trials - m))
