import matplotlib.pyplot as plt
import numpy as np


def e_vs_lambda(e):

    N = e.shape[0]
    n = np.arange(1, N + 1)
    en = np.pi ** 2 * n ** 2

    fig = plt.figure()
    plt.plot(en, e / en, label=fr"$\frac{{\bar{{\lambda}}_n}}{{(\pi n)^2}}$")
    plt.legend(fontsize=15)
    plt.xlabel(r"$\lambda_n = (\pi n)^2$", size=20)
    plt.ylabel(
        fr"$\frac{{\bar{{\lambda}}_n}}{{\lambda_n}}\ /\ [\frac{{2mL^2 E_n}}{{\hbar^2}}]$",
        size=20,
    )
    plt.savefig(f"report/img/e_vs_lambda_N={N}.png", facecolor=fig.get_facecolor())
    plt.show()


def e_vs_n(e):

    N = e.shape[0]
    n = np.arange(1, N + 1)
    en = np.pi ** 2 * n ** 2

    fig = plt.figure()
    plt.plot(n, e / en, label=fr"$\frac{{\bar{{\lambda}}_n}}{{(\pi n)^2}}$")
    plt.legend(fontsize=15)
    plt.xlabel(r"$n$", size=20)
    plt.ylabel(
        fr"$\frac{{\bar{{\lambda}}_n}}{{\lambda_n}}\ /\ [\frac{{2mL^2 E_n}}{{\hbar^2}}]$",
        size=20,
    )
    plt.savefig(f"report/img/e_vs_n_N={N}.png", facecolor=fig.get_facecolor())
    plt.show()
