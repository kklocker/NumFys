import autograd.numpy as np
from autograd import elementwise_grad as egrad
from dask import delayed, compute
from numba import jit
import matplotlib.pyplot as plt

from utils import U, force, rand_gauss, get_time_step, normalized_constants

kb = 1.3806e-23


@jit(nopython=False, forceobj=True)
def euler_scheme(x, n, dt, D):
    """Returns the *next* point in time-domain
    
    Arguments:
        x {float} -- Position at time t
        t {float} -- time
        dt {float} -- time step
    """

    du = force(x, n * dt, 0.2, 1)
    ksi = rand_gauss(x.shape[0])

    return x - du * dt + np.sqrt(2 * D * dt) * ksi


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


if __name__ == "__main__":

    data = {
        "r": 12e-12,
        "L": 20e-6,
        "alpha": 0.2,
        "eta": 1e-3,
        "kbt": 26e-3,
        "delta_u": 80 * 1.60e-19,
    }

    gamma, omega, D = normalized_constants()
    dt = get_time_step(data["alpha"], D, tol=0.1)
    N = 10000  # number of steps

