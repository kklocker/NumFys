import numpy as np  # Array manipulation at C-speed
from numba import njit  # Compile functions at first excecution.


@njit
def get_lattice_pp(N):
    """
    Returns lattice with size NxN,
    and extra rows ++.
    For the Mon-Jasnow algorithm.
    """

    lattice = np.random.choice(np.array([-1, 1]), size=(N + 2, N))  # Original lattice
    lattice[0, :] = np.ones(N)
    lattice[N + 1, :] = np.ones(N)

    # lattice_pp = np.insert(
    # lattice, np.array([0, N]), 1, axis=0
    # )  # Spin up on both edges
    return lattice


@njit
def get_lattice_pm(lat_pp):
    """Creates the +- - equivalent of the ++ lattice-

    """
    tmp = lat_pp.copy()
    tmp[-1, :] *= -1  # Flips the sign of spins at the edge
    return tmp


@njit
def get_flip_energy(i, j, lat, J=1.0):
    """
    (Local Hamilonian)
    Returns integer of sum of nearest spins times the on site spin.
    Assumes periodic boundary conditions in the second axis and no index error in the first axis.
    
    Used for getting the transition probability in metropolis algorithm
    """
    n = lat.shape[1]  # Original number of points along each direction
    sp = lat[i, j]

    j_1 = (j + 1) % n  # PBC y-dir
    j_2 = (j - 1) % n  # PBC y-dir

    sum_other = lat[i - 1, j] + lat[i + 1, j] + lat[i, j_1] + lat[i, j_2]

    return sp * J * sum_other  # * 2  # Difference in energy when flipping
