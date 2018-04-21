import numpy as np
from .distribution import ContinuousDistribution
from ..utils import function

class GaussianDistribution1D(ContinuousDistribution):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

        def probability_density_function(x, mu, sigma):
            mahalanobis_mu_x = (x - mu)**2 / sigma ** 2
            return 1 / (2 * np.pi * sigma ** 2) ** (1/2) *\
                np.exp(- 1 / 2 * mahalanobis_mu_x)
        self.pdf = probability_density_function
        super(GaussianDistribution1D, self).__init__(
                                self.pdf, args=(self.mu, self.sigma))

    @property
    def expected_value(self):
        return function.n_dimension_integral(
            lambda x: x * self.pdf(x, self.mu,
                                   self.sigma), ([-np.inf, np.inf],), 10000)

    @property
    def variance(self):
        e_x_2 = function.n_dimension_integral(
            lambda x: x ** 2 * self.pdf(x, self.mu, self.sigma),
            ([-np.inf, np.inf],), 10000)
        return e_x_2 - self.expected_value ** 2

if __name__ == "__main__":
    gauss = GaussianDistribution1D(mu=0, sigma=1)
    print(gauss.expected_value, gauss.variance)
