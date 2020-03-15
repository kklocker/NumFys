import numpy as np

# import autograd.numpy as np
from autograd import elementwise_grad as egrad
from dask import delayed, compute
from numba import jit, njit
import matplotlib.pyplot as plt
from tqdm import trange
from time import time


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


# def simulate_particle():
#     """
#     Parameters from

#     J. S. Bader, R. W. Hammond, S. A Henck, M. W. Deem, G. A. McDermott, J. M. Bustillo,
#     J. W. Simpson, G. T. Mulhern, and J. M. Rothberg. DNA transport by a micromachined
#     """


# def parallelize_routine():
#     """Routine for parallellizing multiple computations of the problem

#     Arguments:
#         N {integer} -- number of different particles to simulate
#     """


def plot_position_distribution(t_list, avg_pos_list):

    plt.figure(figsize=(10, 10))
    plt.plot(t_list, avg_pos_list)
    # plt.savefig("")
    plt.show()


if __name__ == "__main__":

    data = {
        "r": 12e-12,
        "L": 20e-6,
        "alpha": 0.2,
        "eta": 1e-3,
        "kbt": 1.60 * 26e-22,
        "delta_u": 80 * 1.60e-19,
    }

    alpha = data["alpha"]
    flashing = False
    gamma, omega, D = normalized_constants(**data)
    dt = get_time_step(alpha, D, tol=0.1)
    print(f"Time step: {dt}")

    N_steps = 100000  # number of steps
    N_particles = 1000

    start = time()

    particle_positions = np.zeros(N_particles)
    avg_particle_positions = np.zeros(N_steps)
    for n in range(N_steps):
        f = force(particle_positions, n * dt, alpha, 1.0, flashing)
        particle_positions = euler_scheme(particle_positions, n, dt, D, f)
        avg_particle_positions[n] = np.average(particle_positions)

    stop = time()
    print(f"Total time: {stop-start}")
    t_list = np.linspace(0, N_steps * dt, N_steps)
    plot_position_distribution(t_list, avg_particle_positions)
