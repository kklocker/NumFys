from scipy.sparse import diags
import numpy as np
from numba import jit
from dask import delayed, compute

import utils


@jit
def matrix_box_potential(N, order=4, v=None):
    """Sets up the one-dimensional eigenvalue solver
    
    """
    x = np.ones(N) * N ** 2

    if order == 2:
        d0 = 2 * x
        d1 = -x[:-1]
        if v is not None:
            d0 += v
        A = diags([d1, d0, d1], [-1, 0, 1], dtype=complex)

    elif order == 4:

        d0 = 5 * x / 2
        if v is not None:
            d0 += v
        d1 = -4 * x[:-1] / 3
        d2 = x[:-2] / 12

        A = diags([d2, d1, d0, d1, d2], [-2, -1, 0, 1, 2], dtype=complex)

    elif order == 6:
        d0 = 49 * x / 18
        if v is not None:
            d0 += v
        d1 = -3 * x[:-1] / 2
        d2 = 3 * x[:-2] / 20
        d3 = -x[:-3] / 90

        A = diags([d3, d2, d1, d0, d1, d2, d3], [-3, -2, -1, 0, 1, 2, 3], dtype=complex)

    else:
        raise NotImplementedError

    return A


def solve_eigenvalues(H, N):
    # e, v = eigsh(H, k=(H.shape[0] - 1), which="LM")
    e, v = np.linalg.eigh(H.toarray())

    utils.save_e(e, N)
    utils.save_v(v, N)
    return e, v


def parallellize_solver(Nmin, Nmax, N):
    """Solves N eigenvalue problems with discretization points 1/Nmin to 1/Nmax

    """
    N_list = np.linspace(Nmin, Nmax, N, dtype=int)

    delayed_list = []
    for n in N_list:
        H = delayed(matrix_box_potential)(n)
        res = delayed(solve_eigenvalues)(H, n)
        delayed_list.append(res)

    compute(*delayed_list)
    return


def get_coeffs(Psi_0, v):
    """Gets the coefficient by taking the 
    inner product, implicitly doing the integral.
    
    Arguments:
        Psi_0 {Array} -- [Initial wave (packet)]
        v {Array} -- [description]
    
    Returns:
        [type] -- [description]
    """
    cn = v.T @ Psi_0
    return cn


def psi(e, v, psi_0, t):
    cn = get_coeffs(psi_0, v)
    return (cn * np.exp(-1j * e * t)) @ v


# @jit
def error_metric(psi, psi_analytic):
    """Gives a metric for the error between
    computed and analytic wave-functions
    
    """
    psi_diff = np.abs(psi) - np.abs(psi_analytic)
    e = psi_diff @ psi_diff
    return e


@jit
def errors():
    for N in 10 ** np.linspace(0, 4, dtype=int):
        H = matrix_box_potential(N)
        e, v = solve_eigenvalues(H, N)

        v = np.sqrt(N) * v


@jit
def f(N, v0):
    """Returns the analytic eigenvectors of 
    the Hamiltonian with energies 0<lambda<v0
    
    """
    la = np.linspace(0, v0, N)

    k = np.sqrt(la)
    ka = np.sqrt(v0 - la)
    return (
        np.exp(ka / 3) * (ka * np.sin(k / 3) + k * np.cos(k / 3)) ** 2
        - np.exp(-ka / 3) * (ka * np.sin(k / 3) - k * np.cos(k / 3)) ** 2
    )
