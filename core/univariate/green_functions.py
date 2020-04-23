import pyffs
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from typing import Optional, Union, Tuple
from math.functions import Matern, MissingWendland

plt.style.use('source/custom_style.mplstyle')


class GreenFunction():
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def __init__(self, order: Optional[float] = None, period: float = 2 * np.pi, rtol: float = 1e-3,
                 cutoff: Optional[int] = None, nogibbs: Optional[bool] = None, nogibbs_method: str = 'fejer'):
        self.order = order
        self.period = period
        if cutoff is None:
            self.rtol = rtol
            self.cutoff = self.get_cutoff()
        else:
            self.cutoff = cutoff
            self.rtol = self.get_tol()
        self.bandwidth = 2 * self.cutoff + 1
        self.fourier_coefficients = self._compute_fs_coefficients()
        if nogibbs is None:
            if self.order > 1:
                self.nogibbs = False
            else:
                self.nogibbs = True
        else:
            self.nogibbs = nogibbs
        self.nogibbs_method = nogibbs_method
        self.interpolated_green_function = self.green_function(nogibbs=nogibbs, nogibbs_method=nogibbs_method)

    def get_cutoff(self, min_cutoff: int = 16):
        cutoff = np.fmax(np.ceil((1 / self.rtol) ** (1 / self.order)), min_cutoff)
        composite_cutoff = 2 ** np.ceil(np.log2(cutoff)).astype(int)
        return composite_cutoff

    def get_tol(self):
        return self.cutoff ** (-self.order)

    def _compute_fs_coefficients(self):
        pass

    def green_function(self, nogibbs: bool = False, nogibbs_method: str = 'fejer', resolution: int = 1024):
        if nogibbs:
            ramp = np.abs(np.arange(-self.cutoff, self.cutoff + 1)) / (self.cutoff + 1)
            if nogibbs_method == 'fejer':
                window = 1 - ramp
            elif nogibbs_method == 'sigma-approximation':
                window = np.sinc(ramp)
            else:
                raise ValueError('Argument nogibbs_method must be one of [fejer, sigma-approximation].')
            fs_coefficients = self.fourier_coefficients * window
        else:
            fs_coefficients = self.fourier_coefficients

        if fs_coefficients.size < resolution:
            resolution = np.floor(resolution).astype(int)
            zero_padding = np.zeros(resolution - fs_coefficients.size)
            fs_coefficients = np.concatenate([fs_coefficients, zero_padding], axis=0)

        sampled_green_function = pyffs.iffs(x_FS=fs_coefficients, T=self.period, T_c=self.period / 2,
                                            N_FS=self.bandwidth)
        space_samples = pyffs.ffs_sample(T=self.period, N_FS=self.bandwidth, T_c=self.period / 2,
                                         N_s=sampled_green_function.size)
        interpolated_green_function = interp1d(space_samples, sampled_green_function, kind='cubic',
                                               fill_value='extrapolate')
        return interpolated_green_function

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x % self.period
        return self.interpolated_green_function(x)

    def plot(self, nb_of_periods: int = 3, resolution: int = 1000):
        left = np.floor(nb_of_periods / 2)
        right = nb_of_periods - left
        x_multi_periods = np.linspace(-left * self.period, right * self.period, resolution)
        y_multi_periods = np.real(self.__call__(x_multi_periods))
        x_main_period = np.linspace(0, self.period, resolution)
        y_main_period = np.real(self.__call__(x_main_period))
        plt.plot(x_multi_periods, y_multi_periods, '--', color=self.colors[3], linewidth=2)
        plt.plot(x_main_period, y_main_period, '-', color=self.colors[0], linewidth=2)

    def power_spectrum(self):
        plt.stem(np.arange(-self.cutoff, self.cutoff + 1), np.abs(self.fourier_coefficients), use_line_collection=True)


class GreenExponential(GreenFunction):
    def __init__(self, alpha: Union[complex, float, int], exponent: Union[float, int],
                 period: float = 2 * np.pi, rtol: float = 1e-3, cutoff: Optional[int] = None,
                 nogibbs: Optional[bool] = None, nogibbs_method: str = 'fejer'):
        self.alpha = alpha
        self.exponent = exponent
        order = exponent
        super(GreenExponential, self).__init__(order=order, period=period, rtol=rtol, cutoff=cutoff,
                                               nogibbs=nogibbs, nogibbs_method=nogibbs_method)

    def _compute_fs_coefficients(self):
        frequencies = (2 * np.pi * np.arange(-self.cutoff, self.cutoff + 1) / self.period)
        op_coeffs = (1j * frequencies + self.alpha) ** self.exponent
        fs_coeffs = np.zeros(shape=frequencies.shape, dtype=np.complex128)
        fs_coeffs[op_coeffs != 0] = 1 / op_coeffs[op_coeffs != 0]
        return fs_coeffs


class GreenFractionalDerivative(GreenExponential):
    def __init__(self, exponent: Union[float, int], period: float = 2 * np.pi, rtol: float = 1e-3,
                 cutoff: Optional[int] = None, nogibbs: Optional[bool] = None, nogibbs_method: str = 'fejer'):
        super(GreenFractionalDerivative, self).__init__(alpha=0, exponent=exponent, period=period, rtol=rtol,
                                                        cutoff=cutoff, nogibbs=nogibbs, nogibbs_method=nogibbs_method)


class GreenIteratedDerivative(GreenFractionalDerivative):
    def __init__(self, exponent: int, period: float = 2 * np.pi, rtol: float = 1e-3,
                 cutoff: Optional[int] = None, nogibbs: Optional[bool] = None, nogibbs_method: str = 'fejer'):
        if exponent != np.floor(exponent):
            raise ValueError('Argument `order` must be an integer for object of class GreenIteratedDerivative.')
        super(GreenIteratedDerivative, self).__init__(exponent=exponent, period=period, rtol=rtol, cutoff=cutoff,
                                                      nogibbs=nogibbs, nogibbs_method=nogibbs_method)


class GreenSobolev(GreenFunction):
    def __init__(self, alpha: Union[complex, float, int], exponent: Union[int, float],
                 period: float = 2 * np.pi, rtol: float = 1e-3, cutoff: Optional[int] = None,
                 nogibbs: Optional[bool] = None, nogibbs_method: str = 'fejer'):
        self.alpha = alpha
        self.exponent = exponent
        order = 2 * self.exponent
        super(GreenSobolev, self).__init__(order=order, period=period, rtol=rtol, cutoff=cutoff,
                                           nogibbs=nogibbs, nogibbs_method=nogibbs_method)

    def _compute_fs_coefficients(self):
        frequencies = (2 * np.pi * np.arange(-self.cutoff, self.cutoff + 1) / self.period)
        op_coeffs = (self.alpha + frequencies ** 2) ** self.exponent
        fs_coeffs = np.zeros(shape=frequencies.shape, dtype=np.complex128)
        fs_coeffs[op_coeffs != 0] = 1 / op_coeffs[op_coeffs != 0]
        return fs_coeffs


class GreenFractionalLaplace(GreenSobolev):
    def __init__(self, exponent: float, period: float = 2 * np.pi, rtol: float = 1e-3,
                 cutoff: Optional[int] = None, nogibbs: Optional[bool] = None, nogibbs_method: str = 'fejer'):
        super(GreenFractionalLaplace, self).__init__(alpha=0, exponent=exponent, period=period, rtol=rtol,
                                                     cutoff=cutoff, nogibbs=nogibbs, nogibbs_method=nogibbs_method)


class GreenIteratedBeltrami(GreenFunction):
    def __init__(self, k: np.int, period: float = 2 * np.pi, rtol: float = 1e-3,
                 cutoff: Optional[int] = None, nogibbs: Optional[bool] = None, nogibbs_method: str = 'fejer'):
        self.k = k
        order = 2 * self.k + 1
        super(GreenIteratedBeltrami, self).__init__(order=order, period=period, rtol=rtol, cutoff=cutoff,
                                                    nogibbs=nogibbs, nogibbs_method=nogibbs_method)

    def _compute_fs_coefficients(self):
        frequencies = (2 * np.pi * np.arange(-self.cutoff, self.cutoff + 1) / self.period)
        iterated_ks = (2 * np.pi * np.arange(self.k + 1) / self.period)
        op_coeffs = np.prod(iterated_ks[None, :] ** 2 - frequencies[:, None] ** 2, axis=-1)
        fs_coeffs = np.zeros(shape=frequencies.shape, dtype=np.complex128)
        fs_coeffs[op_coeffs != 0] = 1 / op_coeffs[op_coeffs != 0]
        return fs_coeffs


class GreenBeltrami(GreenFunction):
    def __init__(self, k: int, period: float = 2 * np.pi, rtol: float = 1e-3,
                 cutoff: Optional[int] = None, nogibbs: Optional[bool] = None, nogibbs_method: str = 'fejer'):
        self.k = k
        super(GreenBeltrami, self).__init__(order=2, period=period, rtol=rtol, cutoff=cutoff,
                                            nogibbs=nogibbs, nogibbs_method=nogibbs_method)

    def _compute_fs_coefficients(self):
        frequencies = (2 * np.pi * np.arange(-self.cutoff, self.cutoff + 1) / self.period)
        op_coeffs = self.k ** 2 - frequencies ** 2
        fs_coeffs = np.zeros(shape=frequencies.shape, dtype=np.complex128)
        fs_coeffs[op_coeffs != 0] = 1 / op_coeffs[op_coeffs != 0]
        return fs_coeffs


class GreenMatern(GreenFunction):
    def __init__(self, scale: float, nu: Optional[float] = None, order: Optional[int] = None, period: float = 2 * np.pi,
                 rtol: float = 1e-3,
                 cutoff: Optional[int] = None):
        if nu is None:
            if order not in [2, 4, 6, 8]:
                raise ValueError('Parameter order must be an integer among [2, 4, 6, 8].')
            else:
                self.order = order
                self.nu = (self.order - 1) / 2
        else:
            if nu not in [1 / 2, 3 / 2, 5 / 2, 7 / 2]:
                raise ValueError('Parameter nu must be a half integer among [1/2,3/2,5/2,7/2].')
            else:
                self.nu = nu
            self.order = 1 + 2 * self.nu
        self.scale = scale
        self.period = period
        self.interpolated_green_function = self.green_function()
        super(GreenMatern, self).__init__(order=self.order, period=self.period, rtol=rtol, cutoff=cutoff,
                                          nogibbs=False)

    def green_function(self, nogibbs: bool = False, nogibbs_method: str = 'fejer', resolution: int = 1024):
        return Matern(nu=self.nu, scale=self.scale, period=self.period)

    def _compute_fs_coefficients(self):
        space_samples = pyffs.ffs_sample(T=self.period, N_FS=self.bandwidth, T_c=self.period / 2,
                                         N_s=self.bandwidth)
        sampled_green_function = self.interpolated_green_function(space_samples)
        fs_coeffs = pyffs.ffs(x=sampled_green_function, T=self.period, T_c=self.period / 2, N_FS=self.bandwidth)
        return fs_coeffs


class GreenIteratedMatern(GreenFunction):
    def __init__(self, scale: float, nu: Optional[float] = None, exponent: int = None, order: Optional[int] = None,
                 period: float = 2 * np.pi, rtol: float = 1e-3, cutoff: Optional[int] = None):
        if nu is None:
            if order not in [2, 4, 6, 8]:
                raise ValueError('Parameter order must be an integer among [2, 4, 6, 8].')
            else:
                self.order = exponent * order
                self.nu = (order - 1) / 2
        else:
            if nu not in [1 / 2, 3 / 2, 5 / 2, 7 / 2]:
                raise ValueError('Parameter nu must be a half integer among [1/2,3/2,5/2,7/2].')
            else:
                self.nu = nu
            self.order = exponent * (1 + 2 * self.nu)
        self.exponent = exponent
        self.scale = scale
        self.period = period
        self.base_green_function = Matern(nu=self.nu, scale=self.scale, period=self.period)
        super(GreenIteratedMatern, self).__init__(order=self.order, period=self.period, rtol=rtol, cutoff=cutoff,
                                                  nogibbs=False)

    def _compute_fs_coefficients(self):
        space_samples = pyffs.ffs_sample(T=self.period, N_FS=self.bandwidth, T_c=self.period / 2,
                                         N_s=self.bandwidth)
        sampled_green_function = self.base_green_function(space_samples)
        fs_coeffs = pyffs.ffs(x=sampled_green_function, T=self.period, T_c=self.period / 2, N_FS=self.bandwidth)
        return fs_coeffs ** self.exponent


class GreenWendland(GreenFunction):
    mu_dic = {1 / 2: 2, 3 / 2: 3, 5 / 2: 4, 7 / 2: 5}

    def __init__(self, scale: float, mu_alpha: Optional[Tuple[int, float]] = None, order: Optional[float] = None,
                 period: float = 2 * np.pi, rtol: float = 1e-3, cutoff: Optional[int] = None):
        if mu_alpha is None:
            if order not in [3, 5, 7, 9]:
                raise ValueError('Parameter order must be an integer among [2, 4, 6, 8].')
            else:
                self.order = order
                self.alpha = (self.order / 2) - 1
                self.mu = self.mu_dic[self.alpha]
        else:
            if mu_alpha not in [(2, 1 / 2), (3, 3 / 2), (4, 5 / 2), (5, 7 / 2)]:
                raise ValueError('The tuple mu_alpha must be one of [(2,1/2), (3,3/2), (4,5/2), (5,7/2)].')
            else:
                self.mu, self.alpha = mu_alpha[0], mu_alpha[1]
                self.order = 2 * (self.alpha + 1)
        self.scale = scale
        self.period = period
        self.interpolated_green_function = self.green_function()
        super(GreenWendland, self).__init__(order=self.order, period=self.period, rtol=rtol, cutoff=cutoff,
                                            nogibbs=False)

    def green_function(self, nogibbs: bool = False, nogibbs_method: str = 'fejer', resolution: int = 1024):
        return MissingWendland(mu_alpha=(self.mu, self.alpha), scale=self.scale, period=self.period)

    def _compute_fs_coefficients(self):
        space_samples = pyffs.ffs_sample(T=self.period, N_FS=self.bandwidth, T_c=self.period / 2,
                                         N_s=self.bandwidth)
        sampled_green_function = self.interpolated_green_function(space_samples)
        fs_coeffs = pyffs.ffs(x=sampled_green_function, T=self.period, T_c=self.period / 2, N_FS=self.bandwidth)
        return fs_coeffs


class GreenIteratedWendland(GreenFunction):
    mu_dic = {1 / 2: 2, 3 / 2: 3, 5 / 2: 4, 7 / 2: 5}

    def __init__(self, scale: float, exponent: int, mu_alpha: Optional[Tuple[int, float]] = None,
                 order: Optional[int] = None, period: float = 2 * np.pi, rtol: float = 1e-3,
                 cutoff: Optional[int] = None):
        if mu_alpha is None:
            if order not in [3, 5, 7, 9]:
                raise ValueError('Parameter order must be an integer among [2, 4, 6, 8].')
            else:
                self.order = exponent * order
                self.alpha = (order / 2) - 1
                self.mu = self.mu_dic[self.alpha]
        else:
            if mu_alpha not in [(2, 1 / 2), (3, 3 / 2), (4, 5 / 2), (5, 7 / 2)]:
                raise ValueError('The tuple mu_alpha must be one of [(2,1/2), (3,3/2), (4,5/2), (5,7/2)].')
            else:
                self.mu, self.alpha = mu_alpha[0], mu_alpha[1]
                self.order = 2 * (self.alpha + 1) * exponent
        self.exponent = exponent
        self.scale = scale
        self.period = period
        self.base_green_function = MissingWendland(mu_alpha=(self.mu, self.alpha), scale=self.scale, period=self.period)
        super(GreenIteratedWendland, self).__init__(order=self.order, period=self.period, rtol=rtol, cutoff=cutoff,
                                                    nogibbs=False)

    def _compute_fs_coefficients(self):
        space_samples = pyffs.ffs_sample(T=self.period, N_FS=self.bandwidth, T_c=self.period / 2,
                                         N_s=self.bandwidth)
        sampled_green_function = self.base_green_function(space_samples)
        fs_coeffs = pyffs.ffs(x=sampled_green_function, T=self.period, T_c=self.period / 2, N_FS=self.bandwidth)
        return fs_coeffs ** self.exponent


if __name__ == '__main__':
    # green_function = GreenExponential(alpha=2, exponent=2, nogibbs=True)
    # green_function = GreenWendland(scale=2,mu_alpha=(2, 1 / 2))
    #green_function = GreenIteratedMatern(scale=1, nu=3 / 2, exponent=4)
    green_function = GreenIteratedWendland(scale=1.5,mu_alpha=(2, 1 / 2),exponent=2)
    plt.figure()
    green_function.plot()
    plt.figure()
    green_function.power_spectrum()
