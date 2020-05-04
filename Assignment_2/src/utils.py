import numpy as npp
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from dask import delayed, compute
from numba import jit, njit, vectorize, float64, boolean
import matplotlib.pyplot as plt
import seaborn as sns


# @jit(nopython=False, forceobj=True)
# @njit
def Ur(x, alpha):
    """Help function for the spatial part of the normalized potential
    """
    k1 = 1 / alpha
    k2 = 1 / (1 - alpha)
    a = x.astype(float) % 1.0
    return npp.where(a < alpha, k1 * a, k2 * (1 - a))


# @jit(nopython=True)
# @njit
# @vectorize([float64(float64, float64, boolean)])
@jit(nopython=False, forceobj=True)
def f(t, tau, flashing=True):
    """Help funtion for time-part of potential
    
    Arguments:
        t {array/float} -- time array
        tau {float} -- period
    
    Returns:
        array -- [description]
    """

    l = npp.ones_like(t)
    a = t % tau
    return npp.where(a < 3 * tau / 4, 0 * l, l)


@jit(nopython=False, forceobj=True)
def U(x, t, alpha, tau, flashing=True):
    """Returns  normalized potential

    Returns:
        [type] -- [description]
    """
    if not flashing:
        return Ur(x, alpha)
    return Ur(x, alpha) * f(t, tau)


# @jit(nopython=True)
# @njit
# @vectorize([float64(float64, float64)])
@jit(nopython=False, forceobj=True)
def force_x(x, alpha):
    k1 = 1 / alpha
    k2 = 1 / (1 - alpha)
    a = x % 1.0
    return npp.where(a < alpha, k1, -k2)


# @vectorize([float64(float64, float64, float64, float64, boolean)])
# @njit
@jit(nopython=False, forceobj=True)
def force(x, t, alpha, tau, flashing=True):
    # return -egrad(U, 0)(x, t, alpha, tau, flashing)

    fx = force_x(x, alpha)
    if not flashing:
        return fx
    ft = f(t, tau)
    return fx * ft


@jit(nopython=True)
def rand_gauss(N):
    """Returns a simple gaussian centered around 0 with standard deviation 1
    
    Arguments:
        N {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    return npp.random.normal(0, 1, N)


def get_time_step(alpha, D, tol=0.1):
    """
    Returns a time step that satisfies the criterion for the normalized problem.
    """
    M = npp.max(npp.abs(force(npp.linspace(0, 1, 100), 0.0, alpha, 1.0)))

    dt = 0.1
    while not time_step_condition(dt, M, D) < tol * alpha:
        dt = dt / 2
    return dt


def time_step_condition(dt, M, D):
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
    print(kwargs)
    gamma = 6 * np.pi * eta * r
    omega = delta_u / (gamma * L ** 2)
    D = kbt / delta_u

    return gamma, omega, D


def avg_vel(avg_pos, max_time):
    """Computes the average velocity by taking the difference of the last and first element
     
     Arguments:
         avg_pos  -- [list of average positions for each time step]
         max_time  -- [maximum (NOT NORMALIZED) time of the simulation.]
     """
    return (avg_pos[-1] - avg_pos[0]) / max_time


def save_solution(pos, nc, split_number, particle_type=1, flashing=False):
    N, k = pos.shape
    arr = np.linspace(1, k - 1, split_number, dtype=int)
    tmp_arr = pos[:, arr]
    dt = nc["dt"]
    omega = nc["omega"]
    MT = round(k * dt / omega)
    if flashing:
        np.save(
            f"simulations/particle{particle_type}_flash_N{N}_MT{MT}", [tmp_arr, nc_1]
        )
    else:
        np.save(
            f"simulations/particle{particle_type}_no_flash_N{N}_MT{MT}", [tmp_arr, nc_1]
        )
    print("Saved. ")


def dist_plot(pos, dt, du, omega, N_steps):
    # sns.distplot(
    #     pos[:, 2],
    #     hist=False,
    #     kde=True,
    #     kde_kws={"shade": True},
    #     label=f"$\hat t$ = {2*dt / omega:.4f}s",
    # )

    sns.distplot(
        pos[:, int(N_steps // 2)],
        hist=False,
        kde=True,
        kde_kws={"shade": True},
        label=fr"$\hat t$ = {int(N_steps//2)*dt / omega:.4f}s",
    )

    sns.distplot(
        pos[:, -1],
        hist=False,
        kde=True,
        kde_kws={"shade": True},
        label=fr"$\hat t$ = {N_steps*dt / omega:.4f}s",
    )
    plt.legend(fontsize=15)
    plt.ylabel(f"Density", fontsize=15)
    plt.xlabel(r"$\frac{x}{L}$", fontsize=20)
    plt.title(fr"$\Delta U = {du}\, \mathrm{{k_BT}}$")
    plt.savefig(f"img/du_{du}.png")
    plt.show()
