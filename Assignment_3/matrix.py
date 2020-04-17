from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import numpy as np


def matrix_box_potential(N, order=4):
    """Sets up the one-dimensional eigenvalue solver
    
    """
    x = np.ones(N) * N ** 2

    if order == 2:
        d0 = 2 * x
        d1 = -x[:-1]
        A = diags([d1, d0, d1], [-1, 0, 1], dtype=complex)

    elif order == 4:

        d0 = 5 * x / 2
        d1 = -4 * x[:-1] / 3
        d2 = x[:-2] / 12

        A = diags([d2, d1, d0, d1, d2], [-2, -1, 0, 1, 2], dtype=complex)

    elif order == 6:
        d0 = 49 * x / 18
        d1 = -3 * x[:-1] / 2
        d2 = 3 * x[:-2] / 20
        d3 = -x[:-3] / 90

        A = diags([d3, d2, d1, d0, d1, d2, d3], [-3, -2, -1, 0, 1, 2, 3], dtype=complex)

    else:
        raise NotImplementedError

    return A


def solve_egenvalues(H):
    # e, v = eigsh(H, k=(H.shape[0] - 1), which="LM")
    e, v = np.linalg.eigh(H.toarray())
    np.save("eigsh/eigenvalue", e)
    np.save("eigsh/eigenvectors", v)
    return e, v


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
