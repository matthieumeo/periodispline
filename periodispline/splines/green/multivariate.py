import pyffs
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import RegularGridInterpolator
from typing import Optional, Union, Iterable, Callable
import periodispline.splines.green.univariate as perspline1d

plt.style.use('source/custom_style.mplstyle')


class GreenFunctionND:
    def __init__(self, orders: Union[Iterable, float, None] = None, periods: Union[Iterable, float, None] = None,
                 ndim: int = 2, rtol: float = 1e-3, nogibbs: Optional[bool] = None,
                 cutoffs: Union[Iterable, float, None] = None):
        if ndim < 2:
            raise ValueError(
                'For univariate green functions, use base class periodispline.splines.green.univariate.GreenFunction')
        else:
            self.ndim = ndim
        if periods is None:
            periods = 2 * np.pi
        orders, periods = np.asarray(orders).reshape(-1), np.asarray(periods).reshape(-1)
        if orders.size == self.ndim:
            self.orders = orders
        elif orders.size == 1:
            self.orders = orders * np.ones(shape=(ndim,))
        else:
            raise ValueError('Argument orders should be of size ndim or one.')
        if periods.size == self.ndim:
            self.periods = periods
        elif periods.size == 1:
            self.periods = periods * np.ones(shape=(ndim,))
        else:
            raise ValueError('Argument orders should be of size ndim or one.')
        self.rtol = rtol
        if nogibbs is None:
            self.nogibbs = (np.min(self.orders) < self.ndim)
            self.nogibbs_method = 'fejer' if (np.min(self.orders) < self.ndim) else None
        else:
            self.nogibbs = nogibbs
            self.nogibbs_method = 'fejer' if nogibbs else None
        if cutoffs is None:
            self.cutoffs = self.get_cutoffs()
        else:
            self.cutoffs = np.asarray(cutoffs).reshape(-1)
        self.bandwidths = 2 * self.cutoffs + 1
        self.fourier_coefficients = self._compute_fs_coefficients()
        self.interpolated_green_function = self.green_function()
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def get_cutoffs(self, min_cutoff: int = 16) -> np.ndarray:
        cutoffs = np.clip(np.ceil((1 / self.rtol) ** (1 / self.orders)), a_min=min_cutoff, a_max=None)
        composite_cutoffs = 2 ** np.ceil(np.log2(cutoffs)).astype(int)
        return composite_cutoffs

    def _compute_fs_coefficients(self):
        pass

    def green_function(self, resolution: int = 512) -> Callable:
        if self.ndim > 2:
            resolution = 128
        if self.nogibbs:
            ramps = []
            for i in range(self.ndim):
                ramps.append(1 - np.abs(np.arange(-self.cutoffs[i], self.cutoffs[i] + 1)) / (self.cutoffs[i] + 1))
            meshgrids = np.meshgrid(*ramps)
            windows = np.prod(np.stack(meshgrids, axis=-1), axis=-1)
            fs_coefficients = self.fourier_coefficients * windows
        else:
            fs_coefficients = self.fourier_coefficients
        zero_padding = np.clip(resolution - np.array(fs_coefficients.shape), a_min=1, a_max=None).astype(int)
        padding_list = [(0, padder) for padder in zero_padding]
        fs_coefficients = np.pad(fs_coefficients, padding_list)

        sampled_green_function = fs_coefficients.transpose()
        space_samples = []
        for axis in range(self.ndim):
            sampled_green_function = np.fft.ifftshift(pyffs.iffs(x_FS=sampled_green_function, T=self.periods[axis],
                                                                 T_c=self.periods[axis] / 2,
                                                                 N_FS=self.bandwidths[axis], axis=axis), axes=axis)
            axis_space_samples = np.fft.ifftshift(pyffs.ffs_sample(T=self.periods[axis], N_FS=self.bandwidths[axis],
                                                                   T_c=self.periods[axis] / 2,
                                                                   N_s=sampled_green_function.shape[axis]))
            space_samples.append(axis_space_samples)
        self.space_samples_meshgrid = np.meshgrid(*space_samples)
        self.sampled_green_function = sampled_green_function
        interpolated_green_function = RegularGridInterpolator(points=space_samples, values=sampled_green_function,
                                                              method='linear', bounds_error=False, fill_value=None)
        return interpolated_green_function

    def __call__(self, x: np.ndarray) -> Union[float, np.ndarray]:
        x = x % self.periods
        return self.interpolated_green_function(x)

    def plot(self, nb_of_periods: int = 2, resolution: int = 128, color: Union[int, str, None] = None,
             cmap: Optional[str] = 'RdYlBu_r', plt_type: str = 'wireframe', c: float = 2, a: float = 1,
             ratio: float = 0.6):
        if self.ndim != 2:
            raise NotImplementedError(
                'Plotting not supported for multivariate green functions with dimension greater than two.')
        x = np.linspace(-nb_of_periods * self.periods[0] / 2, nb_of_periods * self.periods[0] / 2, resolution)
        y = np.linspace(-nb_of_periods * self.periods[1] / 2, nb_of_periods * self.periods[1] / 2, resolution)
        X, Y = np.meshgrid(x, y)
        points = np.stack((X, Y), axis=-1)
        Z = np.real(self.__call__(points))
        Z /= np.max(np.abs(Z[np.abs(Z) > 0]))
        if color is None:
            color = self.colors[0]
        elif type(color) is int:
            color = self.colors[color]
        else:
            pass
        if plt_type is 'wireframe':
            fig = plt.figure()
            ax3d = fig.add_subplot(111, projection='3d')
            ax3d.plot_wireframe(X=X, Y=Y, Z=Z, rcount=resolution, ccount=resolution, colors=color, antialiaseds=True,
                                linewidths=0.5)
        elif plt_type is 'plot_surface':
            fig = plt.figure()
            ax3d = fig.add_subplot(111, projection='3d')
            ax3d.plot_surface(X=X, Y=Y, Z=Z, cmap=cmap, rcount=resolution, ccount=resolution, antialiaseds=True)
        elif plt_type is 'pcolormesh':
            fig = plt.figure()
            plt.pcolormesh(X, Y, Z, cmap=cmap, shading='gouraud', snap=True)
            plt.colorbar()
        elif plt_type is 'plotly_surface':
            fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y, colorscale='Plasma', showscale=False, opacity=0.9)])
            # fig.update_traces(contours_z=dict(show=True, usecolormap=True, project_z=True, highlightcolor="limegreen"))
            fig.update_layout(width=1280, height=720,
                              margin=dict(l=0, r=0, b=0, t=0))
            # fig.show(renderer="browser")
            # fig.write_image("exports/fig1.pdf")
        elif plt_type is 'flat_torus':
            x_cart = (c + a * np.cos(X)) * np.cos(Y)
            y_cart = (c + a * np.cos(X)) * np.sin(Y)
            z_cart = a * np.sin(X)
            fig = go.Figure(
                data=[go.Surface(z=z_cart, x=x_cart, y=y_cart, surfacecolor=Z, colorscale='Plasma', showscale=False)])
            fig.update_layout(width=1280, height=720,
                              margin=dict(l=0, r=0, b=0, t=0))
            # fig.show(renderer="browser")
            # fig.write_image("exports/fig1.pdf")
        elif plt_type is 'bump_torus':
            a += ratio * Z / np.max(Z)
            x_cart = (c * a + a * np.cos(X)) * np.cos(Y)
            y_cart = (c * a + a * np.cos(X)) * np.sin(Y)
            z_cart = a * np.sin(X)
            fig = go.Figure(
                data=[go.Surface(z=z_cart, x=x_cart, y=y_cart, surfacecolor=Z, colorscale='Plasma', showscale=False)])
            fig.update_layout(width=1280, height=720,
                              margin=dict(l=0, r=0, b=0, t=0))
            # fig.show(renderer="browser")
        elif plt_type is 'surface_torus':
            fig = make_subplots(rows=1, cols=2,
                                specs=[[{'type': 'surface'}, {'type': 'surface'}]])
            a += ratio * Z
            x_cart = (c * a + a * np.cos(X)) * np.cos(Y)
            y_cart = (c * a + a * np.cos(X)) * np.sin(Y)
            z_cart = a * np.sin(X)
            fig.add_trace(go.Surface(z=Z, x=x, y=y, colorscale='Plasma', showscale=False), row=1, col=1)
            fig.add_trace(
                go.Surface(z=z_cart, x=x_cart, y=y_cart, surfacecolor=Z, colorscale='Plasma', showscale=False), row=1,
                col=2)
            fig.update_layout(width=1280, height=720,
                              margin=dict(l=0, r=0, b=0, t=0),
                              scene=dict(
                                  xaxis=dict(
                                      backgroundcolor="white",
                                      gridcolor="white",
                                      showbackground=False,
                                      zerolinecolor="white", nticks=0, tickfont=dict(color='white')),
                                  yaxis=dict(
                                      backgroundcolor="white",
                                      gridcolor="white",
                                      showbackground=False,
                                      zerolinecolor="white", nticks=0, tickfont=dict(color='white')),
                                  zaxis=dict(
                                      backgroundcolor="white",
                                      gridcolor="white",
                                      showbackground=False,
                                      zerolinecolor="white", nticks=0, tickfont=dict(color='white')),
                                  xaxis_title=' ', yaxis_title=' ', zaxis_title=' '),
                              scene2=dict(
                                  xaxis=dict(
                                      backgroundcolor="white",
                                      gridcolor="white",
                                      showbackground=False,
                                      zerolinecolor="white", nticks=0, tickfont=dict(color='white')),
                                  yaxis=dict(
                                      backgroundcolor="white",
                                      gridcolor="white",
                                      showbackground=False,
                                      zerolinecolor="white", nticks=0, tickfont=dict(color='white')),
                                  zaxis=dict(
                                      backgroundcolor="white",
                                      gridcolor="white",
                                      showbackground=False,
                                      zerolinecolor="white", nticks=0, tickfont=dict(color='white')),
                                  xaxis_title=' ', yaxis_title=' ', zaxis_title=' ')
                              )
            # fig.show(renderer="browser")
        else:
            fig = None
        return fig

    def volume_plot(self, nb_of_periods=2, resolution=32):
        if self.ndim != 3:
            raise NotImplementedError(
                'Volume plots are for trivariate green functions only!')
        x = np.linspace(-nb_of_periods * self.periods[0] / 2, nb_of_periods * self.periods[0] / 2, resolution)
        y = np.linspace(-nb_of_periods * self.periods[1] / 2, nb_of_periods * self.periods[1] / 2, resolution)
        z = np.linspace(-nb_of_periods * self.periods[1] / 2, nb_of_periods * self.periods[1] / 2, resolution)
        X, Y, Z = np.meshgrid(x, y, z)
        points = np.stack((X, Y, Z), axis=-1)
        values = np.real(self.__call__(points))
        values /= np.max(values)
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values.flatten(),
            isomin=np.min(values),
            isomax=1,
            opacity=0.1,  # needs to be small to see through all surfaces
            surface_count=16,  # needs to be a large number for good volume rendering
            colorscale='Plasma',
            showscale=False
        ))
        fig.update_layout(width=1280, height=720,
                          margin=dict(l=0, r=0, b=0, t=0))
        return fig

    def power_spectrum(self, cmap: Optional[str] = 'RdYlBu_r'):
        if self.ndim != 2:
            raise NotImplementedError(
                'Plotting not supported for multivariate green functions with dimension greater than two.')
        freq_u = np.arange(-self.cutoffs[0], self.cutoffs[0] + 1)
        freq_v = np.arange(-self.cutoffs[1], self.cutoffs[1] + 1)
        U, V = np.meshgrid(freq_u, freq_v)
        plt.pcolormesh(U, V, np.abs(self.fourier_coefficients), cmap=cmap, shading='flat', snap=True)
        plt.colorbar()


class GreenSobolevND(GreenFunctionND):
    def __init__(self, alpha: Union[float, int], exponent: Union[float, int],
                 periods: Union[np.ndarray, float, None] = None, ndim: int = 2, rtol: float = 1e-3,
                 nogibbs: Optional[bool] = None):
        self.alpha = alpha
        self.exponent = exponent
        orders = 2 * self.exponent
        super(GreenSobolevND, self).__init__(orders=orders, periods=periods, ndim=ndim, rtol=rtol,
                                             nogibbs=nogibbs)

    def _compute_fs_coefficients(self):
        frequency_lines = []
        for i in range(self.ndim):
            frequency_lines.append(2 * np.pi * np.arange(-self.cutoffs[i], self.cutoffs[i] + 1) / self.periods[i])
        freq_meshgrids = np.stack(np.meshgrid(*frequency_lines), axis=-1)
        op_coeffs = (np.sum(freq_meshgrids ** 2, axis=-1) + self.alpha) ** self.exponent
        fs_coeffs = np.zeros(shape=op_coeffs.shape, dtype=np.complex128)
        fs_coeffs[op_coeffs != 0] = 1 / op_coeffs[op_coeffs != 0]
        return fs_coeffs


class GreenFractionalLaplaceND(GreenSobolevND):
    def __init__(self, exponent: Union[float, int], periods: Union[np.ndarray, float, None] = None, ndim: int = 2,
                 rtol: float = 1e-3, nogibbs: Optional[bool] = None):
        super(GreenFractionalLaplaceND, self).__init__(alpha=0, exponent=exponent, periods=periods, ndim=ndim,
                                                       rtol=rtol, nogibbs=nogibbs)


class GreenSeparableND(GreenFunctionND):
    def __init__(self, *args: perspline1d.GreenFunction, ndim: int = 2):
        if len(args) != ndim:
            raise ValueError(
                f'Not enough Green functions provided for the number of dimensions: {len(args)} != {ndim}.')
        self.UnivariateGreenFunctionsList = args
        orders = []
        periods = []
        rtols = []
        nogibbs = []
        cutoffs = []
        for arg in args:
            orders.append(arg.order)
            periods.append(arg.period)
            rtols.append(arg.rtol)
            nogibbs.append(arg.rtol)
            cutoffs.append(arg.cutoff)
        rtols, nogibbs = np.asarray(rtols), np.asarray(nogibbs)
        rtol = np.min(rtols)
        nogibbs = bool(np.sum(nogibbs).astype(bool))
        super(GreenSeparableND, self).__init__(orders=orders, periods=periods, ndim=ndim, rtol=rtol,
                                               nogibbs=nogibbs, cutoffs=cutoffs)

    def green_function(self, resolution: int = 512) -> Callable:
        return None

    def _compute_fs_coefficients(self):
        univariate_fs_coeffs_list = []
        for gfunc in self.UnivariateGreenFunctionsList:
            univariate_fs_coeffs_list.append(gfunc.fourier_coefficients)
        fs_coeffs = np.prod(np.stack(np.meshgrid(*univariate_fs_coeffs_list), axis=-1), axis=-1)
        return fs_coeffs

    def __call__(self, x: np.ndarray) -> Union[float, np.ndarray]:
        x = x % self.periods
        y = np.ones(shape=x.shape[:-1], dtype=np.complex)
        for i, greenfunc in enumerate(self.UnivariateGreenFunctionsList):
            y *= greenfunc.__call__(x[..., i])
        return y


if __name__ == '__main__':
    green_functionnd = GreenSobolevND(alpha=0, exponent=1, periods=2 * np.pi, ndim=2, nogibbs=True, rtol=1e-3)
    # green_function1 = perspline1d.GreenExponential(alpha=1, exponent=.9, nogibbs=True)
    # green_function2 = perspline1d.GreenSobolev(alpha=10, exponent=3)
    # green_function3 = perspline1d.GreenMatern(scale=0.5, nu=1 / 2)
    # green_functionnd = GreenSeparableND(green_function2, green_function1, ndim=2)
    green_functionnd.plot(plt_type='surface_torus', nb_of_periods=3, resolution=256, ratio=0.5, a=1, c=1.5)
    # green_functionnd.volume_plot(nb_of_periods=2, resolution=32)
    # plt.figure()
    # green_functionnd.power_spectrum()
