from abc import ABCMeta, abstractmethod
import numpy as np
from ..utils import function


class DiscreteDistribution(metaclass=ABCMeta):
    def __init__(self, *probs):
        if not all([0 < prob < 1 for prob in probs]):
            raise ValueError('probability is from 0 to 1')
        if sum(probs) != 1:
            raise ValueError('sum of probability is 1')

    @abstractmethod
    def trial(self):
        pass

    @property
    @abstractmethod
    def expected_value(self):
        pass

    @property
    @abstractmethod
    def variance(self):
        pass


class ContinuousDistribution(metaclass=ABCMeta):
    """
    for 1 dimension continulus distribution

    Parameters
    __________
    args : argument for function
    pdf  : Probability Density Function

    Notes
    _____
    "cdf"
    cumulative distribution function

    "sum_prob"
    sum of probability of x from -inf to inf
    """

    def __init__(self, pdf, n_split=10000, args=None):
        self.essential_figure = 3
        self.pdf = pdf
        self.cdf = lambda x, args, pdf: \
            n_dimension_integral(pdf,
                                ([-np.inf, x],), n_split=n_split, args=args)
        self.sum_prob = self.cdf(np.inf, args=args, pdf=pdf)
        if not np.round(self.sum_prob, self.essential_figure) == 1:
            print("your sum of probalbility is", self.sum_prob)
            raise ValueError("sum of probalbility is 1")

    @property
    @abstractmethod
    def expected_value(self):
        pass

    @property
    @abstractmethod
    def variance(self):
        pass
