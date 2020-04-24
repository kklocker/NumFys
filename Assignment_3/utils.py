import os
import numpy as np
from numba import jit, njit
from scipy.optimize import root
from scipy.linalg import lu_factor, lu_solve

HBAR = 1.05e-34


def get_n_list():
    """Gets a list of all pre-computed eigenvalues
    and eigenvectors based on the number of
    discretization points. 
    """
    n_list = []
    for k in os.walk("eigsh"):
        for split in k[2]:
            n = eval(split.split("_N")[1].split(".npy")[0])
            if n not in n_list:
                n_list.append(n)

    return n_list


def stringbuilder_e(N):
    return f"eigsh/eigenvalues_N{N}"


def stringbuilder_v(N):
    return f"eigsh/eigenvectors_N{N}"


def load_e_v(N):
    e_str = stringbuilder_e(N) + ".npy"
    v_str = stringbuilder_v(N) + ".npy"

    e = np.load(e_str)
    v = np.load(v_str)
    return e, v


def save_e(e, N):
    e_str = stringbuilder_e(N)
    np.save(e_str, e)


def save_v(v, N):
    v_str = stringbuilder_v(N)
    np.save(v_str, v)


@jit
def discretization_error(psi_list, n_list, n_excited=0):
    """
    psi_list: List of nth excited states.
    n_excited: nth energy level
    """
    error = []
    for i, n in enumerate(n_list):
        x = np.linspace(0, 1, psi_list[i].shape[0])
        analytic = np.sqrt(2) * np.sin((n_excited + 1) * np.pi * x)
        err = error_metric(np.sqrt(n) * psi_list[i], analytic)
        error.append(err)

    return error


def error_metric(psi, psi_analytic):
    """Gives a metric for the error between
    computed and analytic wave-functions
    
    """
    psi_diff = np.abs(psi) - np.abs(psi_analytic)
    e = psi_diff @ psi_diff
    # e = simps(psi_diff ** 2, x=np.linspace(0, 1, psi.shape[0]))
    return e


def inner_product(Psi_0, Psi_1):
    """Gets the coefficient by taking the 
    inner product, implicitly doing the integral.
    """
    cn = Psi_1.T.conj() @ Psi_0
    return cn / Psi_0.shape[0]


@njit
def dirac_delta(N):
    """
    Creates a properly normalized delta-distribution at x=1/2.
    """
    psi_delta = np.zeros(N)
    psi_delta[N // 2] = 1
    N = 1 / np.sqrt(inner_product(psi_delta, psi_delta))
    return N * psi_delta


@njit
def box_potential(N, v, vr=0):
    """
    Sets up a box-potential of strength v, discretized by N points
    """
    V = np.zeros(N)
    x = np.linspace(0, 1, N)
    V[(x < (2 / 3)) & (x > (1 / 3))] = v
    if vr != 0:
        V[(x > (2 / 3))] = vr
    return V


@njit
def f(lambda_list, v0):
    """Returns the analytic eigenvectors of
    the Hamiltonian with energies 0<lambda<v0
    """

    k = np.sqrt(lambda_list)
    ka = np.sqrt(v0 - lambda_list)
    return (
        np.exp(ka / 3) * (ka * np.sin(k / 3) + k * np.cos(k / 3)) ** 2
        - np.exp(-ka / 3) * (ka * np.sin(k / 3) - k * np.cos(k / 3)) ** 2
    )


def find_roots(E, v0):
    """
    Finds the roots of the
    function f(lambda) based
    on the computed eigenvalues E
    and an upper limit for the potential v0.
    """
    e = E[E <= v0]
    r0 = np.unique(e.round())
    r = root(f, r0, args=(v0))
    return r["x"]


@njit
def euler_scheme(H, N_temporal, dt, initial_state):
    """
    N_temporal: Number of discretization points in time
    dt: time step
    initial_state: Normalized inital state
    """
    current = initial_state
    for n in range(N_temporal):
        current = current - 1j * dt * (H @ current)
    return current


def crank_nicolson_scheme(H, psi0, dt, N_temporal):
    """
    Sets up LU-decomposition of lhs ans solves a system of equations. 
    Discards intermediate solutions
    """
    N = psi0.shape[0]
    rhs = np.identity(N) - 1j / 2 * dt * H
    lhs = np.identity(N) + 1j / 2 * dt * H

    lu, piv = lu_factor(lhs)
    print("Lu done")
    psi = psi0.copy()

    for n in range(N_temporal):
        b = rhs @ psi
        psi = lu_solve((lu, piv), b)
    return psi


def ev_H(H, vr):
    """ 
    Returns operator for calculating expectation value
    of hamiltonian + vr. 
    
    The matrix element <v1|H+v|v2> is 
    then called by utils.inner_product(op@v2, v1)

    """
    N = H.shape[0]
    x = np.linspace(0, 1, N)
    v = np.zeros(N)
    v[(x > (2 / 3))] = vr

    op = H + v * np.eye(N)
    return op


def two_level_hamiltonian(e, t):
    return np.array([[-e / 2, t], [t, e / 2]])


@njit
def interaction_hamiltonian(ep, omega, tau, t):
    return np.array(
        [
            [0, np.exp(-1j * ep * t / HBAR) * tau * np.sin(omega * t)],
            [np.exp(1j * ep * t / HBAR) * tau * np.sin(omega * t), 0],
        ]
    )


@njit
def p(t, tau):
    return np.sin(t * tau / (2 * HBAR)) ** 2

