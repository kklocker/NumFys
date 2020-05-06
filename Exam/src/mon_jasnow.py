import numpy as np  # Array manipulation at C-speed
from numba import njit  # Compile functions at first excecution.

# from dask import delayed, compute
from lattice_utils import get_flip_energy, get_lattice_pp, get_lattice_pm, energy_diff


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
        for i in range(1, nx - 1):
            for j in range(ny):
                si = lattice[i, j]

                if i == 0:
                    H -= si * (
                        lattice[i, (j - 1) % ny]
                        + lattice[i, (j + 1) % ny]
                        + lattice[i + 1, j]
                    )
                elif i == nx - 1:
                    H -= si * (
                        lattice[i, (j - 1) % ny]
                        + lattice[i, (j + 1) % ny]
                        + lattice[i - 1, j]
                    )

                else:
                    H -= si * (
                        lattice[i, (j - 1) % ny]
                        + lattice[i, (j + 1) % ny]
                        + lattice[(i - 1), j]
                        + lattice[(i + 1), j]
                    )
        return H * J / 2
    elif boundary_cond == "torus":
        for i in range(nx):
            for j in range(ny):
                si = lattice[i, j]

                H -= si * (
                    lattice[i, (j - 1) % ny]
                    + lattice[i, (j + 1) % ny]
                    + lattice[(i - 1) % nx, j]
                    + lattice[(i + 1) % nx, j]
                )
        return H * J / 2
    elif boundary_cond == "klein":

        for i in range(nx):
            for j in range(ny):
                si = lattice[i, j]
                if i == 0:
                    H -= si * (
                        lattice[(i + 1), j]
                        + lattice[i, (j + 1) % ny]
                        + lattice[i, (j - 1) % ny]
                        - lattice[nx - 1, ny - 1 - j]
                    )
                elif i == (nx - 1):
                    H -= si * (
                        lattice[i - 1, j]
                        + lattice[i, (j + 1) % ny]
                        + lattice[i, (j - 1) % ny]
                        - lattice[0, ny - 1 - j]
                    )
                else:
                    H -= si * (
                        lattice[i, (j - 1) % ny]
                        + lattice[i, (j + 1) % ny]
                        + lattice[(i - 1), j]
                        + lattice[(i + 1), j]
                    )
        return H * J / 2


@njit
def metropolis_subroutine(lat_pp, T, bc="mj"):
    """
    Subroutine for metropolis algorithm.
    Handles spin flipping for the lattice. 


    Timings:
          100x100:  14.1 ms ± 339 µs
        1000x1000:  1.56 s ± 40.3 ms

    Returns:
        Nothing. Does all lattice manipulation in place.
    
    """

    N = lat_pp.shape[1]
    n = 0
    while n < N ** 2:  # N^2 Flip tries.
        i, j = np.random.randint(
            0, N, size=2
        )  # Generate random indices and shift in the direction that has extra rows.

        if bc == "mj":  # Shift index to not flip boundary edges
            i += 1

        flip_energy_pp = get_flip_energy(i, j, lat_pp, bc=bc)
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


    EDIT: This function is never really used, but was initially 
    for testing purposes and may safely be ignored

    """
    H_list = np.zeros((N_avg, 2))  # List of tuples for both
    for n in range(N_avg):
        lattice_pp = get_lattice_pp(N_size)
        for _ in range(N_sweeps):
            metropolis_subroutine(lattice_pp, T)  # N_sweeps to obtain equilibrium
        lattice_pm = get_lattice_pm(lattice_pp)
        H_list[n, :] = np.array(
            [ising_hamiltonian(lattice_pp), ising_hamiltonian(lattice_pm)]
        )
    return H_list


@njit
def metropolis_mj_2(N_size, N_sweeps, T, skips=50, N_runs=10, bc="mj"):
    """
    Metropolis algorithm for computing expectation values of energy difference.

    * Invokes the metropolis subroutine
    * Computes energy difference
    * appends to list
    * Compute mean and std.
    """

    ev = []  # List for storing expectation value
    for run in range(N_runs):
        lattice_pp = get_lattice_pp(N_size, aligned=True)  # Create ++ -lattice
        prev = 0  # keeps track of last sampled state
        for i in range(N_sweeps):  # Do N_sweep sweeps

            metropolis_subroutine(lattice_pp, T, bc=bc)  # Tries to flip N^2 spins.

            if (i > max(50, skips)) and (
                i >= (prev + skips)
            ):  # Do at least 50 Initial sweeps, and skip sweeps between each sample
                prev = i  # Keeps track of index of current sample

                E_diff = energy_diff(
                    lattice_pp, bc=bc
                )  # Computes the energy difference in the hamiltonians

                k = np.exp(-E_diff / T)
                # if E_diff < 0 and T < (2 / np.log(1 + np.sqrt(2))):
                # pass
                if (k == 0.0) or k == np.inf:
                    pass
                else:
                    ev.append(k)  # Add exp(-(E_{+-} - E_{++})/T) to a list
    if len(ev) == 0:
        ev.append(
            np.nan
        )  # Just to avoid error computing mean. Happens for large (N > 40) lattices
    ev = np.array(ev)
    return ev.mean(), ev.std() / np.sqrt(N_runs)  # Return the mean and std


@njit
def get_tau(N_size, N_sweeps, T, N_runs=10, skips=10):
    """Returns list of computed tau-values
    
    Arguments:
        N_size {int} -- Size of lattice
        N_sweeps {int} -- Number of sweeps
        T {array} -- T-values
    """
    tau = np.zeros_like(T)
    for i in range(tau.shape[0]):
        mean, _ = metropolis_mj_2(N_size, N_sweeps, T[i], skips=skips, N_runs=N_runs)
        tau[i] = -np.log(mean) * T[i] / N_size
    return tau
