import numpy as np  # Array manipulation at C-speed
from numba import njit  # Compile functions at first excecution.

# from dask import delayed, compute
from lattice_utils import get_flip_energy, get_lattice_pp, get_lattice_pm


@njit
def ising_hamiltonian(lattice, J=1.0, boundary_cond="mj"):
    """Computes the energy of the Ising Hamiltonian of a square lattice. 
    

    Timings:
        NxN:        Time:
            10x10:  2.09 µs ± 67.4 ns 
          100x100:  112 µs ± 9.1 µs 
        1000x1000:  10.1 ms ± 255 µs 

    Arguments: 
        lattice {array} -- Two-dimensional array of a square lattice.

    Keyword Arguments:
        J {float} -- Coupling strength of spins (default: {1.0})
        boundary_cond {str} -- Which boundary conditions to use. 
        Uses periodic by default (default: {None})

    Returns:
        float -- Energy of the lattice.
    """

    H = 0.0
    nx, ny = lattice.shape

    if boundary_cond == "mj":
        for i in range(nx):
            for j in range(ny):
                si = lattice[i, j]

                if i == 0:
                    H -= (
                        J
                        / 2
                        * si
                        * (
                            lattice[i, (j - 1) % ny]
                            + lattice[i, (j + 1) % ny]
                            + lattice[i + 1, j]
                        )
                    )
                elif i == nx - 1:
                    H -= (
                        J
                        / 2
                        * si
                        * (
                            lattice[i, (j - 1) % ny]
                            + lattice[i, (j + 1) % ny]
                            + lattice[i - 1, j]
                        )
                    )

                else:
                    H -= (
                        J
                        / 2
                        * si
                        * (
                            lattice[i, (j - 1) % ny]
                            + lattice[i, (j + 1) % ny]
                            + lattice[(i - 1), j]
                            + lattice[(i + 1), j]
                        )
                    )
        return H
    elif boundary_cond == "torus":
        for i in range(nx):
            for j in range(ny):
                si = lattice[i, j]

                H -= (
                    J
                    / 2
                    * si
                    * (
                        lattice[i, (j - 1) % ny]
                        + lattice[i, (j + 1) % ny]
                        + lattice[(i - 1) % nx, j]
                        + lattice[(i + 1) % nx, j]
                    )
                )
        return H
    else:
        raise NotImplementedError


@njit
def metropolis_subroutine(lat_pp, T, J=1.0):
    """
    Periodic boundary conditions in one direction, 
    extra layers in the other. 

    Timings:
          100x100:  14.1 ms ± 339 µs
        1000x1000:  1.56 s ± 40.3 ms

    Returns:
        Two matrices. Both flipped equally weighted with H_{++}
    
    """

    N = lat_pp.shape[1]
    n = 0
    while n < N ** 2:  # N^2 Flip tries.
        i, j = np.random.randint(0, N, size=2) + np.array(
            [1, 0]
        )  # Generate random indices and shift in the direction that has extra rows.

        flip_energy_pp = get_flip_energy(i, j, lat_pp, J)
        if T == 0.0:
            if flip_energy_pp <= 0.0:
                lat_pp[i, j] *= -1
                # lat_pm[i, j] *= -1
            else:
                pass
        else:
            transition_prob = np.exp(-flip_energy_pp / T)  # W(x|y)
            r = np.random.uniform(0.0, 1.0)

            if transition_prob >= r:
                lat_pp[i, j] *= -1
                # lat_pm[i, j] *= -1
        n += 1


@njit
def metropolis_MJ(N_size, N_sweeps, N_avg, T):
    """
    Computes the metroplolis algorithm For N_avg uncorrelated lattices
    Runs N_sweep times to obtain equilibrium
    returns list of energies for the two (++, +-) -lattices.

    """

    H_list = np.zeros((N_avg, 2))  # List of tuples for both
    for n in range(N_avg):
        lattice_pp = get_lattice_pp(N_size)

        for _ in range(N_sweeps):
            metropolis_subroutine(lattice_pp, T, J=1.0)  # N_sweeps to obtain equilib

        lattice_pm = get_lattice_pm(lattice_pp)

        H_list[n, :] = np.array(
            [ising_hamiltonian(lattice_pp), ising_hamiltonian(lattice_pm)]
        )

    return H_list
