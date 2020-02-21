import scipy.sparse as sp
import scipy.sparse.linalg as spl
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from time import time
import palettable as pl
import os

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
    diag0 = -4 * diag
    diagx = diag[:-1]
    diagy = diag[:-(max_idx)]
    A = sp.diags(
        [diagy, diagx, diag0, diagx, diagy], [-max_idx, -1, 0, 1, max_idx], format="lil"
    )

    fix_matrix(A, max_idx, boundary_dict, classification)
    return A


@jit(nopython=False, forceobj=True)
def fix_matrix(matrix, N_max, boundary_dict=None, contains=None):
    """
    param matrix: Matrix to be "fixed"
    param boundary_dict: Dictionary of boundary points. 
    contains: array of truth values evaluating if a given point is inside or outside the fractal. Precalculated
    """
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


def solve(U, g, pts, k=100, save=True):
    e, v = spl.eigs(U, which="LM", k=k)
    if save:
        pathstr = f"solutions/solution_{g}_{pts}_{k}"
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


if __name__ == "__main__":
    a = time()
    g = 3
    pts = 2
    pos_list = get_grid(g, pts)


    pathstr =f"div/setup_and_solve_{g}_{pts}" 

    if os.path.exists(pathstr):
        time_dict = np.load(pathstr + ".npy", allow_pickle = True).item()
    else:
        time_dict = {}

    b = time()
    print(f"Got poslist. Time: {b-a}")

    m_idx = normalize(pos_list)
    
    c = time()
    print("Max idx: ", m_idx, f"Time: {c-b}\t Total time: {c-a}")
    pos_dict = convert_to_dict(pos_list)

    d = time()
    print(f"Got position_dict. Time: {d-c}\t Total time:  {d-a}")
    classification = contains(pos_dict, m_idx)

    e_ = time()
    print(f"Done classifying. Time: {e_-d}\t Total time: {e_-a}")
    U = get_matrix(m_idx, pos_dict, classification)
    
    f = time()
    print(f"Got matrix. Time: {f-e_}\t Total time: {f-a}")
    e, v = solve(U, g, pts)

    g_ = time()
    print(f"Finished solving. Time: {g_-f}\t Total time: {g_-a}")
    

    time_dict[tuple((g,pts))] = {"classification": e_-d, "matrix_setup":f-e_, "solver":g_-f}

    np.save(pathstr, time_dict)