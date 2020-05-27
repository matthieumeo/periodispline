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

# Fractional Derivative
fig = plt.figure(figsize=(8, 2.5))
exponent = [1, 1.2, 1.4, 1.8, 2, 2.5, 3, 3.5, 4, 6]
weights = np.array([1, -1])
knots = np.array([0, period / 2])
legend = []
# subplot1 = plt.subplot(2, 1, 1)
# subplot2 = plt.subplot(2, 1, 2)

for i, exp in enumerate(exponent):
    green_function = green.GreenFractionalDerivative(exponent=exp, period=period, nogibbs=nogibbs,
                                                     nogibbs_method=nogibbs_method)
    # plt.subplot(subplot1)
    # green_function.plot(nb_of_periods=number_of_periods, resolution=resolution, color=i, linewidth=1.5)
    fractional_spline = UnivariateSpline(weights=weights, knots=knots, green_function=green_function)
    # plt.subplot(subplot2)
    fractional_spline.plot(nb_of_periods=number_of_periods, resolution=resolution, color=i, linewidth=1.5)
    legend.append(f'$\gamma={np.round(exp, 1)}$')

fig.legend(legend, loc='center right')
# plt.suptitle('$D^\gamma$: Green Functions (top), splines (bottom)')
plt.savefig('exports/fractional_derivative.pdf', bbox_inches='tight', transparent=False)

# Fractional Laplacian
fig = plt.figure(figsize=(8, 2.5))
exponent = [1, 1.2, 1.4, 1.8, 2, 2.5, 3, 3.5, 4, 5]
weights = np.array([1, -1])
knots = np.array([0, period / 2])
legend = []
# subplot1 = plt.subplot(2, 1, 1)
# subplot2 = plt.subplot(2, 1, 2)

for i, exp in enumerate(exponent):
    green_function = green.GreenFractionalLaplace(exponent=exp, period=period, nogibbs=nogibbs,
                                                  nogibbs_method=nogibbs_method)
    # plt.subplot(subplot1)
    # green_function.plot(nb_of_periods=number_of_periods, resolution=resolution, color=i, linewidth=1.5)
    fractional_spline = UnivariateSpline(weights=weights, knots=knots, green_function=green_function)
    # plt.subplot(subplot2)
    fractional_spline.plot(nb_of_periods=number_of_periods, resolution=resolution, color=i, linewidth=1.5)
    legend.append(f'$\gamma={np.round(exp, 1)}$')

fig.legend(legend, loc='center right')
# plt.suptitle('$(-\\Delta)^\gamma$: Green Functions (top), splines (bottom)')
plt.savefig('exports/fractional_laplace.pdf', bbox_inches='tight', transparent=False)

# Exponential Spline
fig = plt.figure()
exponent = [1, 1.2, 1.4, 1.8, 2, 2.5, 3, 3.5, 4, 5]
alpha = [1, 3]
legend = []
subplot1 = plt.subplot(2, 1, 1)
subplot2 = plt.subplot(2, 1, 2)
for i, exp in enumerate(exponent):
    plt.subplot(subplot1)
    green_function = green.GreenExponential(exponent=exp, period=period, alpha=alpha[0], nogibbs=nogibbs,
                                            nogibbs_method=nogibbs_method)
    green_function.plot(nb_of_periods=number_of_periods, resolution=resolution, color=i, linewidth=1.5)
    plt.subplot(subplot2)
    green_function = green.GreenExponential(exponent=exp, period=period, alpha=alpha[-1], nogibbs=nogibbs,
                                            nogibbs_method=nogibbs_method)
    green_function.plot(nb_of_periods=number_of_periods, resolution=resolution, color=i, linewidth=1.5)
    legend.append(f'$\gamma={np.round(exp, 1)}$')

fig.legend(legend, loc='center right')
# plt.suptitle(f'$(D+\\alpha Id)^\gamma$: $\\alpha={alpha[0]}$ (top), $\\alpha={alpha[-1]}$ (bottom)')
plt.savefig('exports/exponential_gamma.pdf', bbox_inches='tight', transparent=False)

# Sobolev Operators:
fig = plt.figure()
exponent = np.array([1, 1.2, 1.4, 1.8, 2, 2.5, 3, 3.5, 4, 5])
alpha = [1, 3]
legend = []
subplot1 = plt.subplot(2, 1, 1)
subplot2 = plt.subplot(2, 1, 2)
for i, exp in enumerate(exponent):
    plt.subplot(subplot1)
    green_function = green.GreenSobolev(exponent=exp, period=period, alpha=alpha[0], nogibbs=nogibbs,
                                        nogibbs_method=nogibbs_method)
    green_function.plot(nb_of_periods=number_of_periods, resolution=resolution, color=i, linewidth=1.5)
    plt.subplot(subplot2)
    green_function = green.GreenSobolev(exponent=exp, period=period, alpha=alpha[-1], nogibbs=nogibbs,
                                        nogibbs_method=nogibbs_method)
    green_function.plot(nb_of_periods=number_of_periods, resolution=resolution, color=i, linewidth=1.5)
    legend.append(f'$\gamma={np.round(exp, 1)}$')

fig.legend(legend, loc='center right')
# plt.suptitle(f'$(\\alpha Id-\\Delta)^\gamma$: $\\alpha={alpha[0]}$ (top), $\\alpha={alpha[-1]}$ (bottom)')
plt.savefig('exports/sobolev.pdf', bbox_inches='tight', transparent=False)

# Beltrami Operators:
# fig = plt.figure()
# k_list = np.arange(1, 11)
# weights = np.array([1, -1])
# knots = np.array([0, period / 2])
# legend = []
# subplot1 = plt.subplot(2, 1, 1)
# subplot2 = plt.subplot(2, 1, 2)
# for i, k in enumerate(k_list):
#     plt.subplot(subplot1)
#     green_function = green.GreenIteratedBeltrami(k=k, period=period, nogibbs=nogibbs, nogibbs_method=nogibbs_method)
#     green_function.plot(nb_of_periods=number_of_periods, resolution=resolution, color=i, linewidth=1.5)
#     plt.subplot(subplot2)
#     fractional_spline = UnivariateSpline(weights=weights, knots=knots, green_function=green_function)
#     plt.subplot(subplot2)
#     fractional_spline.plot(nb_of_periods=number_of_periods, resolution=resolution, color=i, linewidth=1.5)
#     legend.append(f'$k={k}$')
#
# fig.legend(legend, loc='center right')
# plt.suptitle('$\Pi_{i=1}^k (D^2+i^2Id)$: Green functions (top), splines (bottom)')
# plt.savefig('exports/iterated_beltrami.pdf', bbox_inches='tight', transparent=False)

# Matern Operators:
fig = plt.figure()
orders = [1, 2, 3, 4]
scale = [1, 0.3]
legend = []
subplot1 = plt.subplot(2, 1, 1)
subplot2 = plt.subplot(2, 1, 2)
for i, order in enumerate(orders):
    plt.subplot(subplot1)
    green_function = green.GreenMatern(scale=scale[0], order=order, period=period)
    green_function.plot(nb_of_periods=number_of_periods, resolution=resolution, color=i, linewidth=1.5)
    plt.subplot(subplot2)
    green_function = green.GreenMatern(scale=scale[-1], order=order, period=period)
    green_function.plot(nb_of_periods=number_of_periods, resolution=resolution, color=i, linewidth=1.5)
    legend.append(f'$\\gamma={green_function.order}$')

fig.legend(legend, loc='center right')
#plt.suptitle(f'Matern: $\\epsilon={scale[0]}$ (top), $\\epsilon={scale[-1]}$ (bottom)')
plt.savefig('exports/matern.pdf', bbox_inches='tight', transparent=False)

# Wendland Operators:
fig = plt.figure()
orders = [3/2, 5/2, 7/2]
scale = [2, 1]
legend = []
subplot1 = plt.subplot(2, 1, 1)
subplot2 = plt.subplot(2, 1, 2)
for i, order in enumerate(orders):
    plt.subplot(subplot1)
    green_function = green.GreenWendland(scale=scale[0], order=order, period=period)
    green_function.plot(nb_of_periods=number_of_periods, resolution=resolution, color=i, linewidth=1.5)
    plt.subplot(subplot2)
    green_function = green.GreenWendland(scale=scale[-1], order=order, period=period)
    green_function.plot(nb_of_periods=number_of_periods, resolution=resolution, color=i, linewidth=1.5)
    legend.append(f'$\\gamma={green_function.order}$')

fig.legend(legend, loc='center right')
#plt.suptitle(f'Wendland: $\\epsilon={scale[0]}$ (top), $\\epsilon={scale[-1]}$ (bottom)')
plt.savefig('exports/wendland.pdf', bbox_inches='tight', transparent=False)
