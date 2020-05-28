import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
import periodispline.splines.green.univariate as green
from periodispline.splines.ndsplines import UnivariateSpline

# Set colors in plots
cmap = get_cmap('tab10')
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[to_hex(c) for c in cmap.colors])

# Setup
period = 2 * np.pi
nogibbs = True
nogibbs_method = 'fejer'
resolution = 1024
number_of_periods = 2

order = [1, 2, 3, 4]
epsilon = 0.5
alpha = 1 / np.sqrt(epsilon)
fig = plt.figure()

for i, ord in enumerate(order):
    sobolev_spline = green.GreenSobolev(exponent=ord, period=period, alpha=alpha, nogibbs=nogibbs,
                                        nogibbs_method=nogibbs_method)
    matern_spline = green.GreenMatern(scale=epsilon, order=ord, period=period)
    plt.subplot(2, 2, i + 1)
    sobolev_spline.plot(nb_of_periods=number_of_periods, resolution=resolution, color=0, linewidth=1.5)
    matern_spline.plot(nb_of_periods=number_of_periods, resolution=resolution, color=1, linewidth=1.5)
    plt.title(f'$\\gamma={ord}$')

fig.legend(['Sobolev', 'Matern'], loc='center right')
plt.suptitle(f'Sobolev vs. Matern splines ($\\epsilon={epsilon}$, $\\alpha={np.round(alpha,2)}$).')
