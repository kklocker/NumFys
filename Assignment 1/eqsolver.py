import scipy.sparse as sp
import scipy.sparse.linalg as spl
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from time import time
import palettable as pl
import os
from dask import delayed, compute
from numba import jit

# Egne filer
from grid import get_grid, normalize, convert_to_dict
from classification import contains, ray_tracing, floodfill


def get_matrix(max_idx, boundary_dict=None, classification=None):
    """ Sets up matrix to solve 2d equation.
    """

    h = 1 / max_idx
    k = max_idx
    n = (k) ** 2  # number of unknowns
    shape = (k, k)
    diag = np.ones(n) / (h ** 2)
    diag0 = 4 * diag
    diagx = -diag[:-1]
    diagy = -diag[:-(max_idx)]
    A = sp.diags(
        [diagy, diagx, diag0, diagx, diagy], [-max_idx, -1, 0, 1, max_idx], format="lil"
    )
    fix_matrix(A, max_idx, boundary_dict, classification)
    return A


def get_matrix_higher_order(M):
    h = 1 / M
    k = M
    n = k ** 2
    diag = np.ones(n) / (12 * h ** 4)
    diag0 = 60 * diag
    diag1 = -16 * diag[:-1]
    diag2 = diag[:-2]
    diagonals = [
        diag2[: -(k - 1)],
        diag1[: -(k - 1)],
        diag2,
        diag1,
        diag0,
        diag1,
        diag2,
        diag1[: -(k - 1)],
        diag2[: -(k - 1)],
    ]
    offsets = [-k - 1, -k, -2, -1, 0, 1, 2, k, k + 1]
    A = sp.diags(diagonals, offsets=offsets, format="lil", shape=(n, n))
    return A


@jit(nopython=False, forceobj=True)
def fix_matrix(matrix, N_max, boundary_dict=None, contains=None, higher_order=False):
    """
    param matrix: Matrix to be "fixed"
    param boundary_dict: Dictionary of boundary points. 
    contains: array of truth values evaluating if a given point is inside or outside the fractal. Precalculated
    """

    if boundary_dict is None:
        boundary_dict = {}
    all_points = [(i, j) for i in range(N_max) for j in range(N_max)]
    # print(len(all_points))

    for idx in range(len(all_points)):
        i, j = all_points[idx]
        if boundary_dict.get(tuple((i, j)), False) or (not contains[i, j]):
            k = i * N_max + j
            k1 = (k + 1) % (N_max ** 2)
            k2 = k - 1
            k3 = (k + N_max) % (N_max ** 2)
            k4 = k - N_max

            matrix[k, k] = 0
            matrix[k, k1] = 0
            matrix[k, k2] = 0
            matrix[k, k3] = 0
            matrix[k, k4] = 0

            matrix[k1, k] = 0
            matrix[k2, k] = 0
            matrix[k3, k] = 0
            matrix[k4, k] = 0

            if higher_order:
                k5 = (k + N_max + 1) % (N_max ** 2)
                k6 = k4 - 1

                matrix[k, k5] = 0
                matrix[k, k6] = 0

                matrix[k5, k] = 0
                matrix[k6, k] = 0


def solve(U, g, pts, k=100, save=True):
    e, v = spl.eigs(U, which="LM", k=k, sigma=238)
    if save:
        pathstr = f"solutions/solution_{g}_{pts}"
        np.save(pathstr, (e, v))
    return e, v


def load_solutions(g, pts, k):
    pathstr = f"solutions/solution_{g}_{pts}_{k}"
    if os.path.exists(pathstr):
        try:
            return np.load(pathstr, allow_pickle=True)
        except Exception as e:
            raise e
    else:
        print("File could not be found. ")
        return


def main_routine(cfg, k=100):
    """ Not properly implemented. Needs a bit of rewriting before these can be run in parallell.
    dask.delayed don't support mutable functions. (Hard to achieve mutation and resilience at the same time)
    """
    # k = 10
    g, pts = cfg
    a = time()
    pos_list = get_grid(g, pts)
    b = time()
    print(f"Getting position list took {b-a}s")
    m_idx = normalize(pos_list)
    c = time()
    print(f"Normalizing took {c-b}s")
    if m_idx <= k:
        # print(m_idx, k)
        k = m_idx - 2  # max eigenvalues. Limitation of ARPACK
    pos_dict = convert_to_dict(pos_list)
    d = time()
    print(f"Converting to dict took {d-c}s")
    c_path = f"solutions/classifications/{g}_{pts}"
    if os.path.exists(c_path + ".npy"):
        classification = np.load(c_path + ".npy")
    else:
        classification = contains(pos_dict, m_idx)
        np.save(c_path, classification)
    e = time()
    print(f"Classification took {e-d}s")
    U = get_matrix(m_idx, pos_dict, classification)
    print(f"Getting matrix took {time()-e}s")
    sol = solve(U, g, pts, k=k)
    # print(f"Finished solving config {cfg}")
    return sol


if __name__ == "__main__":
    a = time()
    g = 4
    pts = 2
    pos_list = get_grid(g, pts)

    pathstr = f"div/setup_and_solve"

    if os.path.exists(pathstr):
        time_dict = np.load(pathstr + ".npy", allow_pickle=True).item()
    else:
        time_dict = {}

    b = time()
    print(f"Got poslist. Time: {b-a:.3f}")

    m_idx = normalize(pos_list)

    c = time()
    print("Max idx: ", m_idx, f"Time: {c-b:.3f}\t Total time: {c-a:.3f}")
    pos_dict = convert_to_dict(pos_list)

    d = time()
    print(f"Got position_dict. Time: {d-c:.3f}\t Total time:  {d-a:.3f}")

    c_path = f"solutions/classifications/{g}_{pts}"
    if os.path.exists(c_path + ".npy"):
        classification = np.load(c_path + ".npy")
    else:
        classification = contains(pos_dict, m_idx)
        np.save(c_path, classification)

    e_ = time()
    print(f"Done classifying. Time: {e_-d:.3f}\t Total time: {e_-a:.3f}")
    U = get_matrix(m_idx, pos_dict, classification)

    f = time()
    print(f"Got matrix. Time: {f-e_:.3f}\t Total time: {f-a:.3f}")
    e, v = solve(U, g, pts)

    g_ = time()
    print(f"Finished solving. Time: {g_-f:.3f}\t Total time: {g_-a:.3f}")

    time_dict[tuple((g, pts))] = {
        "classification": e_ - d,
        "matrix_setup": f - e_,
        "solver": g_ - f,
    }

    np.save(pathstr, time_dict)
