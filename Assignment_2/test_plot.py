import matplotlib.pyplot as plt
import numpy as np

from main import f, Ur, U

if __name__ == "__main__":

    x = np.linspace(-1.5, 1.5, 1000)

    # plt.plot(x, Ur(x, 1, 0.2, 1))
    # plt.show()

    t = np.linspace(0, 10, 1000)

    # plt.plot(t, f(t, 1.0))
    # plt.show()

    X, T = np.meshgrid(x, t)

    plt.contourf(X, T, U(X, T, 1.0, L=1, alpha=0.2, deltaU=1), levels=100)
    plt.colorbar()
    plt.show()

