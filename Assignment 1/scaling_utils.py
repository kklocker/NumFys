import numpy as np


def get_normal(e):
    e = np.sign(e) * e
    e0 = e[0]
    return (e / e0).astype(np.float)


def m(g, p):
    return int((p + 1) / 3 * (5 * 4 ** g - 2))


def trim(arr):
    srted = np.argsort(arr)
    tmp = arr[srted]
    return tmp[tmp >= 0.999]


def integrated_dos(arr):
    tmp = []
    for w in arr:
        count = (arr[arr <= w]).shape[0]
        tmp.append(count)
    return np.array(tmp)


def get_omega_0(eigenvals):  # squared
    integ = []
    e = eigenvals[-1]
    dos = integrated_dos(eigenvals)[-1]

    lim = dos / e
    return 1 / (4 * np.pi * lim)


def get_DN(eigenvals):
    omega = get_omega_0(eigenvals)

    DN = omega / (4 * np.pi) * eigenvals - integrated_dos(eigenvals)
    return DN


def get_lim(idx):
    return integrated_dos(normalized_eigenvals[idx]) / normalized_eigenvals[idx]


def get_mask(arr, c):
    sh = c.shape
    tmp = np.ma.array(arr.reshape(sh), mask=np.invert(c))
    return tmp
