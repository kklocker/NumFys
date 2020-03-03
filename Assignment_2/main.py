import autograd.numpy as np
from autograd import elementwise_grad as egrad

# from dask import Delayed, Compute
from numba import jit
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import square

D = 1


# @jit
def U(x, t):
    """
    Args:
        x: Stochastic function of time
        t: Defines if the potential is on or off

    """
    return (0.5 - np.cos(10 * np.pi * x)) * (0.5 * square(2 * t) + 0.5)


# @jit(nopython=True)
def grad_U(x, t):
    return egrad(U, 0)(x, t)


# @jit
def euler_scheme(x, t, dt):
    """Returns the *next* point in time-domain
    
    Arguments:
        x {float} -- Position at time t
        t {float} -- time
        dt {float} -- time step
    """

    global D

    du = egrad(U, 0)(x, t)
    ksi = np.random.normal(0, 1, x.shape[0])

    return x - du * dt + np.sqrt(2 * D * dt) * ksi


def parallelize_routine(N):
    """Routine for parallellizing multiple computations of the problem
    
    Arguments:
        N {integer} -- number of particles to simulate
    """
    return


def test_plot():
    x = np.linspace(-10, 10)
    t = np.linspace(0, 10)

    X, T = np.meshgrid(x, t)
    u = U(X, T)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.contourf(X, T, u)

    plt.show()


if __name__ == "__main__":

    test_plot()

    n = 1000
    N = 1000
    T = 1

    t = np.linspace(0, T, N)
    dt = T / N
    x = np.zeros(n)
    avg_pos = []
    for i in range(N):
        avg = np.average(x)
        avg_pos.append(avg)
        x = euler_scheme(x, i * dt, dt)

    plt.plot(t, avg_pos)
    plt.show()
