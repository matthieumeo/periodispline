import numpy as np
import periodispline.splines.green.univariate as green
import periodispline.splines.green.multivariate as greenND
from periodispline.splines.ndsplines import MultivariateSpline
import matplotlib.pyplot as plt

plt.style.use('source/custom_style.mplstyle')

# Setup
period = 2 * np.pi
nogibbs = True
resolution = 256
number_of_periods = 2
ndim = 2

# -Delta
knots = np.array([[0, 0], [0, period / 2]])
weights = np.array([1, -1])
green_functionnd = greenND.GreenSobolevND(alpha=0, exponent=1, periods=period, ndim=ndim, nogibbs=True, rtol=1e-3)
spline = MultivariateSpline(weights=weights, knots=knots, green_function=green_functionnd)
fig = spline.plot(nb_of_periods=number_of_periods, resolution=resolution, plt_type='surface_torus', c=1.2, ratio=0.4)
fig.show("browser")
fig.write_image("exports/bidelta.png", width=1920, height=1080, scale=2)

# -Delta^(2)
knots = np.array([[0, 0], [0, period / 2]])
weights = np.array([1, -1])
green_functionnd = greenND.GreenSobolevND(alpha=0, exponent=2, periods=period, ndim=ndim, nogibbs=True, rtol=1e-3)
spline = MultivariateSpline(weights=weights, knots=knots, green_function=green_functionnd)
fig = spline.plot(nb_of_periods=number_of_periods, resolution=resolution, plt_type='surface_torus', c=1.2, ratio=0.4)
fig.show("browser")
fig.write_image("exports/squared_bidelta.png", width=1920, height=1080, scale=2)

# Sobolev
for exp in [1, 2, 3]:
    green_functionnd = greenND.GreenSobolevND(alpha=2, exponent=exp, periods=period, ndim=ndim, nogibbs=True, rtol=1e-3)
    fig = green_functionnd.plot(nb_of_periods=number_of_periods, resolution=resolution, plt_type='surface_torus', c=1.2,
                                ratio=0.4)
    fig.show("browser")
    fig.write_image(f"exports/bisobolev_alpha={green_functionnd.alpha}_exp={green_functionnd.exponent}.png", width=1920,
                    height=1080, scale=2)

# Exp-exp
for exp in [1, 2, 3]:
    green_function1 = green.GreenExponential(alpha=1, exponent=exp, nogibbs=True)
    green_function2 = green.GreenExponential(alpha=3, exponent=exp, nogibbs=True)
    green_functionnd = greenND.GreenSeparableND(green_function1, green_function2, ndim=2)
    fig = green_functionnd.plot(plt_type='surface_torus', nb_of_periods=3, resolution=256, ratio=0.5, a=1, c=1.5)
    fig.show("browser")
    fig.write_image(
        f"exports/exp**2_alpha={green_function1.alpha},{green_function2.alpha}_exp={green_function1.exponent},{green_function2.exponent}.png",
        width=1920, height=1080, scale=2)

# Sobolev-Exp
for exp in [1, 2, 3]:
    green_function1 = green.GreenSobolev(alpha=2, exponent=exp, nogibbs=True)
    green_function2 = green.GreenExponential(alpha=1, exponent=exp, nogibbs=True)
    green_functionnd = greenND.GreenSeparableND(green_function1, green_function2, ndim=2)
    fig = green_functionnd.plot(plt_type='surface_torus', nb_of_periods=3, resolution=256, ratio=0.5, a=1, c=1.5)
    fig.show("browser")
    fig.write_image(
        f"exports/sob_exp_alpha={green_function1.alpha},{green_function2.alpha}_exp={green_function1.exponent},{green_function2.exponent}.png",
        width=1920, height=1080, scale=2)

# Matern-Matern
for nu in [1 / 2, 3 / 2, 5 / 2]:
    green_function1 = green.GreenMatern(scale=1, nu=nu, period=period)
    green_function2 = green.GreenMatern(scale=1, nu=nu, period=period)
    green_functionnd = greenND.GreenSeparableND(green_function1, green_function2, ndim=2)
    fig = green_functionnd.plot(plt_type='surface_torus', nb_of_periods=3, resolution=256, ratio=0.5, a=1, c=1.5)
    fig.show("browser")
    fig.write_image(
        f"exports/mat**2_scale={green_function1.scale},{green_function2.scale}_nu={green_function1.nu},{green_function2.nu}.png",
        width=1920, height=1080, scale=2)

# Wend-Wend
for order in [3/2, 5/2, 7/2]:
    green_function1 = green.GreenWendland(scale=2, order=order, period=period)
    green_function2 = green.GreenWendland(scale=2, order=order, period=period)
    green_functionnd = greenND.GreenSeparableND(green_function1, green_function2, ndim=2)
    fig = green_functionnd.plot(plt_type='surface_torus', nb_of_periods=3, resolution=256, ratio=0.5, a=1, c=1.5)
    fig.show("browser")
    fig.write_image(
        f"exports/wend**2_scale={green_function1.scale},{green_function2.scale}_order={green_function1.order},{green_function2.order}.png",
        width=1920, height=1080, scale=2)

# Wend-Matern

for order in [3, 5, 7]:
    green_function1 = green.GreenWendland(scale=2, order=order, period=period)
    green_function2 = green.GreenMatern(scale=1, order=order - 1, period=period)
    green_functionnd = greenND.GreenSeparableND(green_function1, green_function2, ndim=2)
    fig = green_functionnd.plot(plt_type='surface_torus', nb_of_periods=3, resolution=256, ratio=0.5, a=1, c=1.5)
    fig.show("browser")
    fig.write_image(
        f"exports/wend_matern_scale={green_function1.scale},{green_function2.scale}_order={green_function1.order},{green_function2.order}.png",
        width=1920, height=1080, scale=2)
