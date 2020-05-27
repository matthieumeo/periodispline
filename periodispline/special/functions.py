import numpy as np
from typing import Tuple, Union


class Matern:
    def __init__(self, nu: float, scale: float, period: float = 2 * np.pi):
        if nu not in [1 / 2, 3 / 2, 5 / 2, 7 / 2]:
            raise ValueError('Parameter nu must be a half integer among [1/2,3/2,5/2,7/2].')
        else:
            self.nu = nu
        self.scale = scale
        self.period = period
        self.max = self.matern_function(0)

    def r(self, t: np.ndarray) -> np.ndarray:
        return np.sqrt(2 - 2 * np.cos(2 * np.pi * t / self.period))

    def matern_function(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if self.nu == (1 / 2):
            return np.exp(-r)
        elif self.nu == (3 / 2):
            return (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
        elif self.nu == (5 / 2):
            return (1 + np.sqrt(5) * r + 5 * (r ** 2) / 3) * np.exp(-np.sqrt(5) * r)
        elif self.nu == (7 / 2):
            return (1 + np.sqrt(7) * r + 42 * (r ** 2) / 15
                    + 7 * np.sqrt(7) * (r ** 3) / 15) * np.exp(-np.sqrt(7) * r)

    def __call__(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.matern_function(self.r(t) / self.scale) / self.max


class MissingWendland:
    def __init__(self, mu_alpha: Tuple[int, float], scale: float, period: float = 2 * np.pi, zero=1e-12):
        if mu_alpha not in [(2, 1 / 2), (3, 3 / 2), (4, 5 / 2)]:
            raise ValueError('The tuple mu_alpha must be one of [(2,1/2), (3,3/2), (4,5/2)].')
        else:
            self.mu_alpha = mu_alpha
        self.scale = scale
        self.period = period
        self.zero = zero
        self.max = self.missing_wendland_function(self.zero)

    def r(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.sqrt(2 - 2 * np.cos(2 * np.pi * t / self.period))

    def S(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.sqrt(1 - r ** 2)

    def L(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.log(r / (1 + self.S(r)))

    # def P(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    #     return 3465 * (r ** 12) + 83160 * (r ** 10) + 13860 * (r ** 8)
    #
    # def Q(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    #     return 37495 * (r ** 10) + 160290 * (r ** 8) + 33488 * (r ** 6) - 724 * (r ** 4) + 1344 * (r ** 2) - 128

    def missing_wendland_function(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if self.mu_alpha == (2, 1 / 2):
            return 3 * (r ** 2) * self.L(r) + (2 * (r ** 2) + 1) * self.S(r)
        elif self.mu_alpha == (3, 3 / 2):
            return -(15 * r ** 6 + 90 * r ** 4) * self.L(r) - (81 * r ** 4 + 28 * r ** 2 - 4) * self.S(r)
        elif self.mu_alpha == (4, 5 / 2):
            return (945 * r ** 8 + 2520 * r ** 6) * self.L(r) + (
                    256 * r ** 8 + 2639 * r ** 6 + 690 * r ** 4 - 136 * r ** 2 + 16) * self.S(r)
        # elif self.mu_alpha == (5, 7 / 2):
        #     return -self.P(r) * self.L(r) - self.Q(r) * self.S(r)

    def __call__(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        t = np.asarray(t)
        r = self.r(t)
        if r.ndim == 0:
            r = self.zero
        else:
            r[r <= self.zero] = self.zero
        y = np.zeros(r.shape)
        y[r <= self.scale] = self.missing_wendland_function(r[r <= self.scale] / self.scale) / self.max
        return y


if __name__ == '__main__':
    func = MissingWendland(mu_alpha=(3, 3 / 2), scale=2)
