# import numpy as np
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from dask import delayed, compute
from numba import jit
import matplotlib.pyplot as plt


@jit(nopython=False)
def Ur(x, alpha):
    """Help function for the spatial part of the normalized potential
    """
    k1 = 1 / alpha
    k2 = 1 / (1 - alpha)
    a = x % 1.0
    return np.where(a < alpha, k1 * a, k2 * (1 - a))


@jit(nopython=False)
def f(t, tau, flashing=True):
    """Help funtion for time-part of potential
    
    Arguments:
        t {array/float} -- time array
        tau {float} -- period
    
    Returns:
        array -- [description]
    """

    l = np.ones_like(t)
    if not flashing:
        return l
    a = t % tau
    return np.where(a < 3 * tau / 4, 0 * l, l)


@jit(nopython=False, forceobj=True)
def U(x, t, alpha, tau, flashing=True):
    """Returns  normalized potential

    Returns:
        [type] -- [description]
    """
    return Ur(x, alpha) * f(t, tau, flashing)


# @jit(nopython=False)
def force(x, t, alpha, tau):
    return -egrad(U, 0)(x, t, alpha, tau)


def rand_gauss(N):
    """Returns a simple gaussian centered around 0 with standard deviation 1
    
    Arguments:
        N {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    return np.random.normal(0, 1, N)


def get_time_step(alpha, D, tol=0.1):
    """
    Returns a time step that satisfies the criterion for the normalized problem.
    """
    M = np.max(np.abs(force(np.linspace(0, 1, 100), 0.0, alpha, 1.0)))

    dt = 0.1
    while not g(dt, M, D) < tol * alpha:
        dt = dt / 2
    return dt


@jit(nopython=False)
def g(dt, M, D):
    """
    Help function for determining time step.
    """
    return M * dt + 4 * np.sqrt(2 * D * dt)


def normalized_constants(**kwargs):
    """
    Normalizes constants of the problem
    """
    r = kwargs.get("r")
    L = kwargs.get("L")
    eta = kwargs.get("eta")
    kbt = kwargs.get("kbt")
    delta_u = kwargs.get("delta_u")

    gamma = 6 * np.pi * eta * r
    omega = delta_u / (gamma * L ** 2)
    D = kbt / delta_u

    return gamma, omega, D
