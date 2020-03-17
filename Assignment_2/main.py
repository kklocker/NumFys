import numpy as np

# import autograd.numpy as np
from autograd import elementwise_grad as egrad
from dask import delayed, compute
from numba import jit, njit
import matplotlib.pyplot as plt
from tqdm import trange
from time import time
import seaborn as sns

sns.set()


from utils import U, force, rand_gauss, get_time_step, normalized_constants

kb = 1.3806e-23


@njit  # (nopython=False, forceobj=True)
def euler_scheme(x, n, dt, D, F):
    # du = force(x, n * dt, 0.2, 1, flashing)

    ksi = rand_gauss(x.shape[0])
    # print(F.shape)
    return x - F * dt + np.sqrt(2 * D * dt) * ksi


def boltzmann(U, du, kbT):
    """
    Returns a boltzmann distribution from U
    
    Arguments:
        U {[type]} -- [description]
        du {[type]} -- [description]
        KbT {[type]} -- [description]
    """
    return np.exp(-(U * du) / kbT) / (kbT * (1 - np.exp(-(du / kbT))))


def plot_position_distribution(t_list, avg_pos_list, std):

    plt.figure(figsize=(10, 10))
    plt.plot(t_list, avg_pos_list)
    plt.show()


@jit(nopython=False, forceobj=True)
def solution_loop(N_steps, N_particles, dt, alpha, D, tau, flashing):

    particle_positions = np.zeros((N_particles, N_steps))

    for n in range(1, N_steps):
        f = force(particle_positions[:, n - 1], n * dt, alpha, tau, flashing)
        particle_positions[:, n] = euler_scheme(
            particle_positions[:, n - 1], n, dt, D, f
        )
    return particle_positions


# @jit(nopython=False, forceobj=True)
def particles_simulation(data, N_steps, N_particles, flashing=False, tau=None):
    """Runs a simulation of particles specified from "data"
    
    Arguments:
        data {dict} -- Dictionary containing all relevant data of the particle
        N_steps {int} -- Number of time steps
        N_particles {int} -- Number of noninteracting particles in the ensamble
    """

    alpha = data["alpha"]
    gamma, omega, D = normalized_constants(**data)
    dt = get_time_step(alpha, D, tol=0.05)

    particle_positions = solution_loop(
        N_steps, N_particles, dt, alpha, D, tau, flashing
    )

    nc = {"gamma": gamma, "omega": omega, "D": D, "dt": dt}
    return particle_positions, nc


def particles_simulation2(time, N_particles, tau=None, flashing=False):
    """Runs a simultaion of particles over a fixed time interval with
     different values of Delta U
     time: Seconds!
    """

    data = {
        "r": 12e-9,
        "L": 20e-6,
        "alpha": 0.2,
        "eta": 1e-3,
        "kbt": 26 * 1.60217662e-22,
    }
    du_factor = [0.1, 0.5, 1.0, 5.0, 10.0]  # Ratio to the thermal energy

    for du in du_factor:
        data["delta_u"] = du * data["kbt"]
        alpha = data["alpha"]
        gamma, omega, D = normalized_constants(**data)
        ntime = omega * time
        dt = get_time_step(data["alpha"], D, tol=0.08)
        N_steps = int(ntime // dt)
        print(f"Number of steps: {N_steps}")
        ppos = solution_loop(N_steps, N_particles, dt, alpha, D, tau, flashing)

        np.save(
            f"simulations/du_factor_{du}",
            [ppos, {"gamma": gamma, "omega": omega, "D": D, "dt": dt}],
        )


def particles_simulation_flashing():

    data1 = {
        "r": 12e-9,
        "L": 20e-6,
        "alpha": 0.2,
        "eta": 1e-3,
        "kbt": 26 * 1.60e-22,
        "delta_u": 80 * 1.60e-19,
    }

    data2 = {
        "r": 3 * 12e-9,  # Three times the radius
        "L": 20e-6,
        "alpha": 0.2,
        "eta": 1e-3,
        "kbt": 26 * 1.60e-22,
        "delta_u": 80 * 1.60e-19,
    }


def load_simulations_du(du, N_steps, N_particles):
    pos, nc = np.load(
        f"simulations/DU_{du}_{N_steps}_{N_particles}.npy", allow_pickle=True
    )

    return pos, nc


if __name__ == "__main__":

    data = {
        "r": 12e-9,
        "L": 20e-6,
        "alpha": 0.2,
        "eta": 1e-3,
        "kbt": 26 * 1.60e-22,
        "delta_u": 80 * 1.60e-19,
    }

    N_particles = 1000

    # particles_simulation2(5.0, N_particles, None, False)

