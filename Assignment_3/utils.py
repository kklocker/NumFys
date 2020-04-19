import os
import numpy as np


def get_n_list():
    """Gets a list of all pre-computed eigenvalues
    and eigenvectors based on the number of
    discretization points. 
    
    """
    n_list = []
    for k in os.walk("eigsh"):
        for split in k[2]:
            n = eval(split.split("_N")[1].split(".npy")[0])
            if n not in n_list:
                n_list.append(n)

    return n_list


def stringbuilder_e(N):
    return f"eigsh/eigenvalues_N{N}"


def stringbuilder_v(N):
    return f"eigsh/eigenvectors_N{N}"


def load_e_v(N):
    e_str = stringbuilder_e(N) + ".npy"
    v_str = stringbuilder_v(N) + ".npy"

    e = np.load(e_str)
    v = np.load(v_str)
    return e, v


def save_e(e, N):
    e_str = stringbuilder_e(N)
    np.save(e_str, e)


def save_v(v, N):
    v_str = stringbuilder_v(N)
    np.save(v_str, v)
