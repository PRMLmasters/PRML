from abc import ABCMeta, abstractmethod


class DiscreteDistribution(metaclass=ABCMeta):
    def __init__(self, *probs):
        if not all([0 < prob < 1 for prob in probs]):
            raise ValueError('probability is from 0 to 1')
        elif sum(probs) != 1:
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
