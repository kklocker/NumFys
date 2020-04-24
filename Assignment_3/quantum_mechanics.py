from scipy.sparse import diags
from scipy.integrate import simps
from scipy.linalg import lu_factor, lu_solve
import numpy as np
from numba import jit
from dask import delayed, compute

import utils
from plot_functions import *


class WaveFunction:
    def __init__(self, N, V=None):
        if V is not None:
            try:
                if len(V) != N:
                    print("The potential must be discretized in N points!")
                    raise RuntimeError
            except Exception as e:
                print(e)
                raise e

        self.N = N  # discretization points
        self.V = V  # Potential
        self.psi_constructed = False
        self.potential_strength = max(V) if V is not None else None
        self.initial_state_set = False

    def construct_hamiltonian(self, order=4):
        """Sets up the one-dimensional eigenvalue 
        discrete Hamiltonian (with or without a potential)
        
        """
        N = self.N
        v = self.V
        x = np.ones(N) * N ** 2

        if order == 2:
            d0 = 2 * x
            d1 = -x[:-1]
            if v is not None:
                d0 += v
            H = diags([d1, d0, d1], [-1, 0, 1], dtype=complex)

        elif order == 4:

            d0 = 5 * x / 2
            if v is not None:
                d0 += v
            d1 = -4 * x[:-1] / 3
            d2 = x[:-2] / 12

            H = diags([d2, d1, d0, d1, d2], [-2, -1, 0, 1, 2], dtype=complex)

        elif order == 6:
            d0 = 49 * x / 18
            if v is not None:
                d0 += v
            d1 = -3 * x[:-1] / 2
            d2 = 3 * x[:-2] / 20
            d3 = -x[:-3] / 90

            H = diags(
                [d3, d2, d1, d0, d1, d2, d3], [-3, -2, -1, 0, 1, 2, 3], dtype=complex
            )

        else:
            raise NotImplementedError

        self.H = H

    def solve_eigenvalues(self):
        # e, v = eigsh(H, k=(H.shape[0] - 1), which="LM")
        e, v = np.linalg.eigh(self.H.toarray())

        utils.save_e(e, self.N)
        utils.save_v(v, self.N)
        self.e = e  # Eigenvalues
        self.v = v  # Eigenvectors

    def parallellize_solver(self, Nmin, Nmax, N):
        """Solves N eigenvalue problems
        with discretization points 1/Nmin to 1/Nmax
        Used for solving many systems and investigation of
        how the problem scale as the discretization
        step is changeda
        """
        N_list = np.linspace(Nmin, Nmax, N, dtype=int)

        delayed_list = []
        for n in N_list:
            H = delayed(self.construct_hamiltonian)(n)
            res = delayed(self.solve_eigenvalues)(H, n)
            delayed_list.append(res)

        compute(*delayed_list)
        return

    def construct_psi(self, psi_0, t):
        """
        Creates the full wavefunction as a linear combination of stationary states.
        """
        v = np.sqrt(self.N) * self.v
        cn = utils.inner_product(psi_0, v)
        self.cn = cn
        self.psi = (cn * np.exp(-1j * self.e * t[:, None])) @ v.T
        self.psi_constructed = True

    def set_initial_state(self, psi0):

        N = utils.inner_product(psi0, psi0)
        self.psi0 = psi0 / (np.sqrt(N))
        print(
            f"Initial state set. Normalization: {utils.inner_product(self.psi0, self.psi0)}"
        )
        self.initial_state_set = True

    def euler(self, N_temporal, dt):
        """
        Invokes the Euler-scheme for computing the time evolution of an initial state
        """
        if not self.initial_state_set:
            print("Initial state not set!")
            raise RuntimeError
        psi = utils.euler_scheme(self.H.toarray(), N_temporal, dt, self.psi0)
        self.psi_euler = psi

    def factor_lu(self):
        self.lu = lu_factor(self.H.toarray())

    def crank_nicolson(self, N_temporal, dt):
        """
        Invokes the Crank-Nicolson for computing the time evolution of an initial state. 
        """

        self.psi_crank = 

    def __repr__(self):
        return f"{type(self)}\nN: {self.N}, Constructed psi: {self.psi_constructed}"
