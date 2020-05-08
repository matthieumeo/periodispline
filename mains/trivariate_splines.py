import numpy as np
import periodispline.splines.green.univariate as green
import periodispline.splines.green.multivariate as greenND
from periodispline.splines.ndsplines import MultivariateSpline

# Setup
period = 2 * np.pi
nogibbs = True
resolution = 48
number_of_periods = 2
ndim = 3

# -Delta
knots = np.array([[0, 0, 0], [0, period / 2, period / 2]])
weights = np.array([1, -1])
green_functionnd = greenND.GreenSobolevND(alpha=0, exponent=1, periods=period, ndim=ndim, nogibbs=True, rtol=1e-3)
spline = MultivariateSpline(weights=weights, knots=knots, green_function=green_functionnd)
fig = spline.volume_plot(nb_of_periods=number_of_periods, resolution=resolution)
fig.show("chrome")
fig.write_image("exports/tridelta_.png", width=1920, height=1080, scale=2)

# -Delta^(2)
knots = np.array([[0, 0, 0], [0, period / 2, period / 2]])
weights = np.array([1, -1])
green_functionnd = greenND.GreenSobolevND(alpha=0, exponent=2, periods=period, ndim=ndim, nogibbs=True, rtol=1e-3)
spline = MultivariateSpline(weights=weights, knots=knots, green_function=green_functionnd)
fig = spline.volume_plot(nb_of_periods=number_of_periods, resolution=resolution)
fig.show("chrome")
fig.write_image("exports/squared_tridelta_.png", width=1920, height=1080, scale=2)

# Sobolev
for exp in [1, 2, 3]:
    green_functionnd = greenND.GreenSobolevND(alpha=2, exponent=exp, periods=period, ndim=ndim, nogibbs=True, rtol=1e-3)
    fig = green_functionnd.volume_plot(nb_of_periods=number_of_periods, resolution=resolution)
    fig.show("chrome")
    fig.write_image(f"exports/sobolev_alpha={green_functionnd.alpha}_exp={green_functionnd.exponent}.png", width=1920,
                    height=1080, scale=2)

# Exp-exp-exp
for exp in [1, 2, 3]:
    green_function1 = green.GreenExponential(alpha=.5, exponent=exp, nogibbs=True, rtol=1e-2)
    green_function2 = green.GreenExponential(alpha=1, exponent=exp, nogibbs=True, rtol=1e-2)
    green_function3 = green.GreenExponential(alpha=1.5, exponent=exp, nogibbs=True, rtol=1e-2)
    green_functionnd = greenND.GreenSeparableND(green_function1, green_function2, green_function3, ndim=3)
    fig = green_functionnd.volume_plot(nb_of_periods=number_of_periods, resolution=resolution)
    fig.show("chrome")
    fig.write_image(
        f"exports/exp**3_alpha={green_function1.alpha},{green_function2.alpha},{green_function3.alpha}_exp={green_function1.exponent},{green_function2.exponent},{green_function3.exponent}.png",
        width=1920, height=1080, scale=2)

# Sobolev-Sobolev-Sobolev
for exp in [1, 2, 3]:
    green_function1 = green.GreenSobolev(alpha=2, exponent=exp, nogibbs=True, rtol=1e-2)
    green_function2 = green.GreenSobolev(alpha=2, exponent=exp, nogibbs=True, rtol=1e-2)
    green_function3 = green.GreenSobolev(alpha=2, exponent=exp, nogibbs=True, rtol=1e-2)
    green_functionnd = greenND.GreenSeparableND(green_function1, green_function2, green_function3, ndim=3)
    fig = green_functionnd.volume_plot(nb_of_periods=number_of_periods, resolution=resolution)
    fig.show("chrome")
    fig.write_image(
        f"exports/sob**3_alpha={green_function1.alpha},{green_function2.alpha},{green_function3.alpha}_exp={green_function1.exponent},{green_function2.exponent},{green_function3.exponent}.png",
        width=1920, height=1080, scale=2)

# Matern-Matern-matern
for nu in [1 / 2, 3 / 2, 5 / 2]:
    green_function1 = green.GreenMatern(scale=1, nu=nu, period=period, rtol=1e-2)
    green_function2 = green.GreenMatern(scale=1, nu=nu, period=period, rtol=1e-2)
    green_function3 = green.GreenMatern(scale=1, nu=nu, period=period, rtol=1e-2)

    green_functionnd = greenND.GreenSeparableND(green_function1, green_function2, green_function3, ndim=3)
    fig = green_functionnd.volume_plot(nb_of_periods=number_of_periods, resolution=resolution)
    fig.show("chrome")
    fig.write_image(
        f"exports/matern**3_nu={green_function1.nu},{green_function2.nu},{green_function3.nu}_scale={green_function1.scale},{green_function2.scale},{green_function3.scale}.png",
        width=1920, height=1080, scale=2)

# Wednland**3

for order in [3, 5, 7]:
    green_function1 = green.GreenWendland(scale=2, order=order, period=period, rtol=1e-2)
    green_function2 = green.GreenWendland(scale=2, order=order, period=period, rtol=1e-2)
    green_function3 = green.GreenWendland(scale=2, order=order, period=period, rtol=1e-2)

    green_functionnd = greenND.GreenSeparableND(green_function1, green_function2, green_function3, ndim=3)
    fig = green_functionnd.volume_plot(nb_of_periods=number_of_periods, resolution=resolution)
    fig.show("chrome")
    fig.write_image(
        f"exports/wendland**3_order={green_function1.order},{green_function2.order},{green_function3.order}_scale={green_function1.scale},{green_function2.scale},{green_function3.scale}.png",
        width=1920, height=1080, scale=2)
