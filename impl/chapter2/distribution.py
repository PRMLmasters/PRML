from abc import ABCMeta, abstractmethod


class Distribution(metaclass=ABCMeta):

    @abstractmethod
    def trial(self):
        pass

    @abstractmethod
    def expected_value(self):
        pass

    @abstractmethod
    def variance(self):
        pass
