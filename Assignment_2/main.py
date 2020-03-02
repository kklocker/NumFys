import autograd.numpy as np
from autograd import elementwise_grad as egrad
from dask import Delayed, Compute
from numba import jit


def U(x, t):
    """
    Args:
        x: Stochastic function of time
        t: Defines if the potential is on or off

    """
    return


def parallelize_routine(N):
    """Routine for parallellizing multiple computations of the problem
    
    Arguments:
        N {integer} -- number of particles to simulate
    """


if __name__ == "__main__":
    pass
