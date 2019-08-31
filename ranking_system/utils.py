from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import scipy.stats

Parameter = TypeVar('Parameter', np.ndarray, float)


@dataclass
class GaussianDistribution:
    mean: Parameter = None
    precision: Parameter = None

    @property
    def variance(self):
        assert self.precision > 0
        return 1. / self.precision

    @property
    def natural_mean(self):
        return self.precision * self.mean

    @property
    def std(self):
        return np.sqrt(self.variance)

    def update_mean_and_precision(self,
                                  updated_mean=None,
                                  updated_pre=None,
                                  updated_nat_mean=None,
                                  updated_var=None
                                  ):
        assert not (updated_pre is None and updated_var is None)
        if updated_var is None:
            self.precision = updated_nat_mean / updated_mean if updated_pre is None else updated_pre
        else:
            self.precision = 1. / updated_var
        self.mean = updated_nat_mean / self.precision if updated_mean is None else updated_mean

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return scipy.stats.norm.pdf(x, loc=self.mean, scale=self.std)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return scipy.stats.norm.cdf(x, loc=self.mean, scale=self.std)


@dataclass
class StandardGaussian(GaussianDistribution):
    mean: Parameter = 0.
    precision: Parameter = 1.

    def psi(self, x: np.ndarray) -> np.ndarray:
        return self.pdf(x) / self.cdf(x)

    def lambda_(self, x: np.ndarray) -> np.ndarray:
        return self.psi(x) * (self.psi(x) + x)




