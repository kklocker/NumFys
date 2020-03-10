# import numpy as np
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from dask import delayed, compute
from numba import jit
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import square as sq


D = 1


square = jit(nopython=False)(sq)


def Ur(x, L, alpha, deltaU):
    k1 = deltaU / (alpha)
    k2 = 1 / (1 - alpha)
    a = x % 1.0
    return np.where(a < alpha, k1 * a, k2 * (1 - a))


def f(t, tau):
    l = np.ones_like(t)
    a = t % tau
    return np.where(a < 3 * tau / 4, 0 * l, l)


@jit(nopython=False)
def U(x, t, tau, L=20e-6, alpha=0.2, deltaU=80 * 1.6e-19):
    return Ur(x, L, alpha, deltaU) * f(t, tau)


@jit
def force(x, t):
    return -egrad(U, 0)(x, t)


@jit
def euler_scheme(x, t, dt):
    """Returns the *next* point in time-domain
    
    Arguments:
        x {float} -- Position at time t
        t {float} -- time
        dt {float} -- time step
    """

    global D

    du = egrad(U, 0)(x, t, 1.0)
    ksi = gaussian(x.shape[0])

    return x - du * dt + np.sqrt(2 * D * dt) * ksi


@jit(nopython=True)
def gaussian(N):
    return np.random.normal(0, 1, N)


def simulate_particle():
    """
    Parameters from

    J. S. Bader, R. W. Hammond, S. A Henck, M. W. Deem, G. A. McDermott, J. M. Bustillo,
    J. W. Simpson, G. T. Mulhern, and J. M. Rothberg. DNA transport by a micromachined
    """
    r1 = 12e-12
    L = 20e-6
    alpha = 0.2
    eta = 1e-3
    kbt = 26e-3
    delta_u = 80 * 1.60e-19


def parallelize_routine():
    """Routine for parallellizing multiple computations of the problem
    
    Arguments:
        N {integer} -- number of different particles to simulate
    """


def test_plot():
    x = np.linspace(-10, 10)
    t = np.linspace(0, 10)

    X, T = np.meshgrid(x, t)
    u = U(X, T, 1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.contourf(X, T, u)

    plt.show()


if __name__ == "__main__":

    test_plot()

    # n = 1000
    # N = 1000
    # T = 1

    # t = np.linspace(0, T, N)
    # dt = T / N
    # x = np.zeros(n)
    # avg_pos = []
    # for i in range(N):
    #     avg = np.average(x)
    #     avg_pos.append(avg)
    #     x = euler_scheme(x, i * dt, dt)

    # plt.plot(t, avg_pos)
    # plt.show()
