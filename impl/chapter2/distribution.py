from abc import ABCMeta, abstractmethod


class DiscreteDistribution(metaclass=ABCMeta):
    def __init__(self, prob):
        if not 0 <= prob <= 1:
            raise ValueError('probability is from 0 to 1')

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
