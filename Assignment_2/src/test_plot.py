import matplotlib.pyplot as plt
import numpy as np

from utils import U


if __name__ == "__main__":

    x = np.linspace(-1.5, 1.5, 1000)

    # plt.plot(x, Ur(x, 1, 0.2, 1))
    # plt.show()

    t = np.linspace(0, 10, 1000)

    # plt.plot(t, f(t, 1.0))
    # plt.show()

    X, T = np.meshgrid(x, t)

    plt.contourf(X, T, U(X, T, alpha=0.2, tau=1.0, flashing=True), levels=100)
    plt.colorbar()
    plt.show()
