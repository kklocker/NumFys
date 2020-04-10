import numpy as np

# import autograd.numpy as np
from autograd import elementwise_grad as egrad
from dask import delayed, compute
from numba import jit, njit
import matplotlib.pyplot as plt
from tqdm import trange
from time import time
import seaborn as sns
from scipy.integrate import simps

sns.set()


from utils import U, force, rand_gauss, get_time_step, normalized_constants

kb = 1.3806e-23


# @njit  # (nopython=False, forceobj=True)
def euler_scheme(x, dt, D, F):
    # du = force(x, n * dt, 0.2, 1, flashing)

    ksi = rand_gauss(x.shape[0])

    return x - F * dt + np.sqrt(2 * D * dt) * ksi


def boltzmann(U, du, kbT):
    """
    Returns a boltzmann distribution from U
    
    Arguments:
        U {[type]} -- [description]
        du {[type]} -- [description]
        KbT {[type]} -- [description]
    """
    print(du, kbT)
    f = du / kbT
    print(f)
    return np.exp(-(U * f)) / (kbT * (1 - np.exp(-(f))))


def particle_diffusion(x, t, N, particle):
    """Solution for particle density
    
    Arguments:
        x  -- [position]
        t -- [time]
    """

    kbt = particle["kbt"]
    gamma = 6 * np.pi * particle["eta"] * particle["r"]
    D = kbt / gamma
    print(D)
    return (N / (np.sqrt(4 * np.pi * D * t))) * np.exp(-(x ** 2) / (4 * D * t))


def normalize_distribution(dist):
    """Normalizes a distribution
    
    """
    N = simps(dist, dx=1 / dist.shape[0])
    return dist / N


def plot_position_distribution(t_list, avg_pos_list, std):

    plt.figure(figsize=(10, 10))
    plt.plot(t_list, avg_pos_list)
    plt.show()


@jit(nopython=False, forceobj=True)
def solution_loop(N_steps, N_particles, dt, alpha, D, tau, flashing, potential_on=True):

    particle_positions = np.zeros((N_steps, N_particles))
    curr_pos = np.zeros(N_particles)
    for n in range(1, N_steps):
        if potential_on:
            f = force(curr_pos, n * dt, alpha, tau, flashing)
        else:
            f = np.zeros_like(curr_pos)
        curr_pos = euler_scheme(curr_pos, dt, D, f)
        particle_positions[n] = curr_pos
    return particle_positions


@jit(nopython=False, forceobj=True)
def solution_loop_average(N_steps, N_particles, dt, alpha, D, tau, flashing):
    """ Returns average positions of simulation, and the individual particle positions at det last time step.
    
    Returns:
        Array, Array -- average positions, last step positions
    """
    particle_positions = np.zeros(N_steps)
    curr_pos = np.zeros(N_particles)
    for n in range(1, N_steps):
        f = force(curr_pos, n * dt, alpha, tau, flashing)
        curr_pos = euler_scheme(curr_pos, dt, D, f)
        particle_positions[n] = np.average(curr_pos)
    return particle_positions, curr_pos


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


def particles_simulation_3(
    data, max_time, N_particles, tau=0.0, flashing=False, potential_on=True
):
    """Runs a simulation of particles specified from "data"
    
    Arguments:
        data {dict} -- Dictionary containing all relevant data of the particle
        max_time -- Maximum time. Number of steps will be deduced.
        N_particles {int} -- Number of noninteracting particles in the ensamble
    """
    alpha = data["alpha"]
    gamma, omega, D = normalized_constants(**data)
    ntime = omega * max_time
    dt = get_time_step(alpha, D, tol=0.08)
    N_steps = int(ntime // dt)
    particle_positions = solution_loop(
        N_steps, N_particles, dt, alpha, D, omega * tau, flashing, potential_on
    )

    nc = {"gamma": gamma, "omega": omega, "D": D, "dt": dt}
    return particle_positions, nc


def average_particle_simulation(data, max_time, N, tau, flashing=True):

    gamma, omega, D = normalized_constants(**data)
    ntime = omega * max_time

    print(f"gamma: {gamma}, \tomega: {omega},\t D: {D}")
    dt = get_time_step(data["alpha"], D, tol=0.08)
    N_steps = int(ntime // dt)
    print(f"dt: {dt}, \t N_steps: {N_steps}")
    avg_pos, last_step = solution_loop_average(
        N_steps, N, dt, data["alpha"], D, omega * tau, flashing
    )

    nc = {"gamma": gamma, "omega": omega, "D": D, "dt": dt, "N_steps": N_steps}
    return avg_pos, nc, last_step


def tau_simulation(data, N, tau_list, flashing=True):
    avg_pos_list = []
    last_step_list = []
    nc_list = []

    max_times = [max(5, 3 * t) for t in tau_list]

    gamma, omega, D = normalized_constants(**data)

    ntime_list = [omega * max_time for max_time in max_times]

    print(f"gamma: {gamma}, \tomega: {omega},\t D: {D}")
    dt = get_time_step(data["alpha"], D, tol=0.08)
    N_steps_list = [int(ntime // dt) for ntime in ntime_list]
    print(f"dt: {dt}, \t N_steps: {N_steps_list}")
    for i, tau in enumerate(tau_list):
        a = time()
        avg_pos, last_step = solution_loop_average(
            N_steps_list[i], N, dt, data["alpha"], D, omega * tau, flashing
        )

        avg_pos_list.append(avg_pos)
        last_step_list.append(last_step)
        b = time()
        print(f"Loop took {b-a}s. Loop number: {i}")
        nc = {
            "gamma": gamma,
            "omega": omega,
            "D": D,
            "dt": dt,
            "N_steps": N_steps_list[i],
        }
        nc_list.append(nc)
    return avg_pos_list, nc_list, last_step_list


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
