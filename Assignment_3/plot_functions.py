import matplotlib.pyplot as plt
import numpy as np
import palettable as pl
from matplotlib import rc


st = plt.style.available[23]
plt.style.use(st)
emr = pl.cartocolors.sequential.Emrld_7.get_mpl_colormap()
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc("font", **{"family": "serif", "serif": ["Palatino"]})
rc("text", usetex=True)
rc("xtick", labelsize=14)
rc("ytick", labelsize=14)
rc("legend", fontsize=18)


def e_vs_lambda(e):

    N = e.shape[0]
    n = np.arange(1, N + 1)
    en = np.pi ** 2 * n ** 2

    fig = plt.figure(figsize=(6, 3), tight_layout=True)
    plt.plot(en, e / en, label=fr"$\bar\lambda_n$")
    plt.legend(fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r"$\lambda_n = (\pi n)^2$", size=20)
    plt.ylabel(
        fr"$\bar E /\ [\frac{{2mL^2 E_n}}{{\hbar^2}}]$", size=20,
    )
    plt.savefig(f"report/img/e_vs_lambda_N={N}.png", facecolor=fig.get_facecolor())
    plt.show()


def e_vs_n(e):

    N = e.shape[0]
    n = np.arange(1, N + 1)
    en = np.pi ** 2 * n ** 2

    fig = plt.figure(figsize=(6, 3), tight_layout=True)
    plt.plot(n, e / en, label=fr"$\bar\lambda_n$")
    plt.legend(fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r"$n$", size=20)
    plt.ylabel(
        fr"$\bar E / [\frac{{2mL^2 E_n}}{{\hbar^2}}]$", size=20,
    )
    plt.savefig(f"report/img/e_vs_n_N={N}.png", facecolor=fig.get_facecolor())
    plt.show()


def plot_wavefunction(psi):
    plt.figure(constrained_layout=True)
    x = np.linspace(0, 1, psi.shape[0])
    plt.plot(x, np.abs(psi) ** 2)
    plt.show()


def plot_eigenstates(wave, num, V=None, save_name=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.tick_params(axis="y")
    ax.set_ylabel(fr"$|\bar\psi|^2$", size=20)
    ax.set_xlabel(r"$x$", size=20)

    x = np.linspace(0, 1, wave.v[:, 0].shape[0])
    for i in range(num):
        ax.plot(
            x, np.abs(np.sqrt(wave.N) * wave.v[:, i]) ** 2, label=fr"$\bar\psi_{i}$"
        )
    ax.legend(fontsize=12)

    if V is not None:
        ax2 = ax.twinx()
        ax2.plot(x, V / max(V), "--", lw=1.2, color="r")
        ax2.grid()
        ax2.tick_params(axis="y", labelcolor="r")
        ax2.set_ylim(0, 1.5)
        ax2.set_ylabel(fr"$\frac{{\nu(x')}}{{\nu_0}}$", size=20, color="r")
        ax2.yaxis.labelpad = 12
    fig.tight_layout()
    if save_name is not None:
        plt.savefig(f"report/img/{save_name}.png", facecolor=fig.get_facecolor())
