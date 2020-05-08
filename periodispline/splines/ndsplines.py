import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from periodispline.splines.green.univariate import GreenFunction
from periodispline.splines.green.multivariate import GreenFunctionND
from typing import Union, Optional
from plotly.subplots import make_subplots

plt.style.use('source/custom_style.mplstyle')


class UnivariateSpline:

    def __init__(self, weights: np.ndarray, knots: np.ndarray, green_function: GreenFunction):
        if weights.shape != knots.shape:
            raise ValueError('The parameters weights and knots must have the same shape.')
        if isinstance(green_function, GreenFunction) is False:
            raise TypeError(f'Attribute green_function must be of type:{GreenFunction}')
        self.weights = weights
        self.knots = knots
        self.green_function = green_function
        self.period = self.green_function.period
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def __call__(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        t_shifts = (t[:, None] - self.knots[None, :])
        spline = np.sum(self.green_function(t_shifts) * self.weights[None, :], axis=-1)
        return spline

    def plot(self, nb_of_periods: int = 3, resolution: int = 1024, color=None, linewidth=2):
        x = np.linspace(-nb_of_periods * self.period / 2, nb_of_periods * self.period / 2, resolution)
        y = np.real(self.__call__(x))
        y /= np.max(y)
        if color is None:
            plt.plot(x, y, '-', color=self.colors[0], linewidth=linewidth)
        elif type(color) is int:
            plt.plot(x, y, '-', color=self.colors[color], linewidth=linewidth)
        else:
            plt.plot(x, y, '-', color=color, linewidth=linewidth)


class MultivariateSpline:

    def __init__(self, weights: np.ndarray, knots: np.ndarray, green_function: GreenFunctionND):
        if weights.size != knots.shape[0]:
            raise ValueError('The shapes of the parameters weights and knots are incompatible.')
        if isinstance(green_function, GreenFunctionND) is False:
            raise TypeError(f'Attribute green_function must be of type:{GreenFunctionND}')
        self.weights = weights
        self.knots = knots
        self.green_function = green_function
        self.ndim = green_function.ndim
        self.periods = self.green_function.periods
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def __call__(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        t_shifts = (t[..., None, :] - self.knots.reshape((1,) * self.ndim + self.knots.shape))
        spline = np.sum(
            self.green_function(t_shifts) * self.weights.reshape((1,) * self.ndim + self.weights.shape),
            axis=-1)
        return spline

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
            #fig.show(renderer="browser")
            # fig.write_image("exports/fig1.pdf")
        elif plt_type is 'flat_torus':
            x_cart = (c + a * np.cos(X)) * np.cos(Y)
            y_cart = (c + a * np.cos(X)) * np.sin(Y)
            z_cart = a * np.sin(X)
            fig = go.Figure(
                data=[go.Surface(z=z_cart, x=x_cart, y=y_cart, surfacecolor=Z, colorscale='Plasma', showscale=False)])
            fig.update_layout(width=1280, height=720,
                              margin=dict(l=0, r=0, b=0, t=0))
            #fig.show(renderer="browser")
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
            #fig.show(renderer="browser")
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
            #fig.show(renderer="browser")
        else:
            fig = None
        return fig

    def volume_plot(self, nb_of_periods=2, resolution=32):
        if self.green_function.ndim != 3:
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
        return fig
