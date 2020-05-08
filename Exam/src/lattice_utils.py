import numpy as np  # Array manipulation at C-speed
from numba import njit  # Compile functions at first excecution.


@njit
def basic_lattice(N, aligned=True):
    """ 
    Returns a square lattice in an initial state.
    Default is full magnetization with spin up on all sites
    """

    if aligned:
        return np.ones((N, N), dtype=np.float_)
    else:
        return np.random.choice(np.array([-1.0, 1.0]), size=(N, N))


@njit
def get_lattice_pp(N, aligned=False):
    """
    Returns lattice with size NxN,
    and extra rows ++.
    For the Mon-Jasnow algorithm.
    """
    if aligned:
        lattice = np.ones((N + 2, N), dtype=np.float_)
    else:
        lattice = np.random.choice(np.array([-1.0, 1.0]), size=(N + 2, N))
        lattice[0, :] = np.ones(N, dtype=np.float_)
        lattice[N + 1, :] = np.ones(N, dtype=np.float_)
    return lattice


@njit
def convert_pp_to_pm(lat_pp):
    """
    Creates the +- - equivalent of the ++ lattice-

    """
    tmp = lat_pp.copy()
    tmp[-1, :] *= -1  # Flips the sign of spins at the edge
    return tmp


@njit
def get_lattice_pm(N):
    """
    Get (+-) - lattice
    """
    lattice = np.random.choice(np.array([-1.0, 1.0]), size=(N + 2, N))
    lattice[0, :] = np.ones(N, dtype=np.float_)
    lattice[N + 1, :] = -np.ones(N, dtype=np.float_)
    return lattice


@njit
def get_flip_energy(i, j, lat, bc="mj"):
    """
    (Local Hamilonian)
    Returns integer of sum of nearest spins times the on site spin.
    Assumes periodic boundary conditions in the second axis and no index error in the first axis.
    
    Used for getting the transition probability in metropolis algorithm
    """

    sp = lat[i, j]

    J = 1.0
    if bc == "mj":  # Boundary conditions for the original Mon Jasnow Algorithm
        n = lat.shape[1]  # Original number of points along each direction

        j_1 = (j + 1) % n  # PBC y-dir
        j_2 = (j - 1) % n  # PBC y-dir

        if i == 1:
            l1 = lat[i - 1, j] / 2
        else:
            l1 = lat[i - 1, j]

        if i == lat.shape[0] - 2:
            l2 = lat[i + 1, j] / 2
        else:
            l2 = lat[i + 1, j]

        sum_other = l1 + l2 + lat[i, j_1] + lat[i, j_2]

        return (
            sp * J * sum_other * 2
        )  # Difference in energy when flipping. Minussign in H accounted for

    elif bc == "torus":  # Boundary condition for torus

        nx, ny = lat.shape
        sum_other = (
            lat[(i - 1) % nx, j]
            + lat[(i + 1) % nx, j]
            + lat[i, (j + 1) % ny]
            + lat[i, (j - 1) % ny]
        )

        return sp * J * sum_other * 2


@njit
def energy_diff(l, bc="mj"):
    """
    Returns either the energy difference 2m_s = H_+- - H-++
    as in equation (2) in the paper, or the difference Hk - Ht
    l is the lattice.

    This is to save time by not having to compute the Hamiltonian ( which is of order N^2) 


    Some timings for bc = 'torus:

    N:      Diff by Hamiltonian:        This        Speedup factor
    1000    18  ms                      3 µs        6000
    100     205 µs                    899 ns        228
    30      55  µs                    728 ns        75
    """
    if bc == "mj":
        return np.sum(l[-2, :]) * 2  # Original Mon -Jasnow

    elif bc == "torus":
        return np.sum(l[-1, :] * (l[0, ::-1] + l[0, :]))
    else:
        raise NotImplementedError
