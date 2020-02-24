import numpy as np
from dask import delayed, compute
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import palettable as pl
from grid import get_grid, normalize
from numba import jit
import os


@jit(nopython=False, forceobj=True)
def ray_tracing(point, pos_dict, max_idx):
    """
    Should add support for dictionary containing all previously found points. 
    This might, however, be memory heavy.
    """
    if pos_dict.get(tuple(point), False):
        return True

    # sign = 1
    x0, y0 = point

    a = np.min([abs(max_idx - x0), x0])
    b = np.min([abs(max_idx - y0), y0])
    # print(a, b)
    if a < b:
        idx = 0
    else:
        idx = 1

    other_idx = (idx + 1) % 2
    # print(idx, other_idx)
    sign = np.sign(point[idx] - max_idx // 2)
    if sign == 0:
        sign = 1
    # print("SIGN", sign)
    neighbour_sign = None
    p = list(point)
    inside = False
    #     print(p,inside, sign)
    it = 0
    while (p[idx] > 0) & (p[idx] < max_idx):
        prev_p = p.copy()
        check = pos_dict.get(tuple(p), False)  # check if point is on boundary
        temp = p[idx] + sign
        p[idx] = temp
        new_check = pos_dict.get(tuple(p), False)
        # print(check, new_check)
        if new_check and (not check):
            tmp1 = p.copy()
            tmp2 = p.copy()
            tmp1[other_idx] += 1
            tmp2[other_idx] -= 1
            up = pos_dict.get(tuple(tmp1), False)
            down = pos_dict.get(tuple(tmp2), False)
            if up & down:
                inside = not inside
            elif up:
                neighbour_sign = 1
                # print("Sign set to 1")
            elif down:
                neighbour_sign = -1
                # print("Sign set to -1")
            else:
                # print(f"Her burde vi ikke være. p = {p}, idx= {idx}, other_idx = {other_idx}")
                pass
        elif check and (not new_check):
            tmp1 = prev_p.copy()
            tmp2 = prev_p.copy()
            tmp1[other_idx] += 1
            tmp2[other_idx] -= 1
            up = pos_dict.get(tuple(tmp1), False)
            down = pos_dict.get(tuple(tmp2), False)

            if up and down:
                inside = not inside
            elif up == (neighbour_sign == 1):
                pass
            elif down == (neighbour_sign == -1):
                pass
            else:
                inside = not inside

        else:
            pass
        it += 1
        # print(p)
        if it > max_idx:
            print(point)
            print(it)
            break
    if pos_dict.get(tuple(p), False):
        inside = True
    return inside


def floodfill(matrix, start_point, pos_dict):
    """ Reaches maximum recursion depth, but 
    should in principle work by choosing a starting point in the middle
    of the fractal, which we know is inside. 
    """
    i, j = start_point
    print(start_point)
    N = matrix.shape[0]
    if not pos_dict.get(tuple(start_point), False):
        matrix[i, j] = True

        if i > 0:
            floodfill(matrix, tuple((i - 1, j)), pos_dict)
            # print(i)
        if i < (N - 1):
            floodfill(matrix, tuple((i + 1, j)), pos_dict)
        if j > 0:
            floodfill(matrix, tuple((i, j - 1)), pos_dict)
        if j < (N - 1):
            floodfill(matrix, tuple((i, j + 1)), pos_dict)


def contains(pos_dict, max_idx, f=None, parallell=False):
    """
    boundary_dict: dictionary of boundary points 
    max_idx: largest index of square array
    f: Function for determining the 
    """
    pts = np.array(
        [(i, j) for i in range(max_idx) for j in range(max_idx)]
    )  # all possible points in our square
    # pts = [[106,106]]

    temp = []
    if parallell:

        if f == None:  # use matplotlib.path
            pos_list = np.array(list(pos_dict))
            path = mpath.Path(pos_list)

            for p in pts:
                c = delayed(path.contains_point)(p)
                temp.append(c)
        else:
            for p in pts:
                c = delayed(f)(p, pos_dict, max_idx)
                temp.append(c)

        contains = np.array(compute(*temp)).reshape((max_idx, max_idx))
    else:
        if f == None:  # use matplotlib.path
            pos_list = np.array(list(pos_dict))
            path = mpath.Path(pos_list)
            contains = path.contains_points(pts).reshape((max_idx, max_idx))
        else:
            for p in pts:
                c = f(p, pos_dict, max_idx)
                temp.append(c)
            contains = np.array(temp).reshape((max_idx, max_idx))

    return contains


if __name__ == "__main__":
    g = 3  # recursion depth
    pts = 2  # points between each corner of the fractal
    pos_list = get_grid(g, pts)
    m_idx = normalize(pos_list)
    pos_dict = {tuple(p): True for p in pos_list}

    p = False
    f = None

    frac = np.zeros((m_idx, m_idx))
    a = time()

    # floodfill(frac, tuple((m_idx//2, m_idx//2)),pos_dict)
    # point = [106,106]
    # co = contains(pos_dict, m_idx, parallell=p, f=ray_tracing)
    c = contains(pos_dict, m_idx, parallell=p, f=f)
    b = time()
    print(f"Time: {b-a} with g={g}, pts = {pts}, parallell = {p}")
    frac[c] += 1
    emr = pl.cartocolors.sequential.Emrld_7.get_mpl_colormap()
    # plt.imshow(frac, cmap="Paired")
    # plt.show()

    plot_list = [*pos_list, pos_list[0]]

    plt.style.use("Solarize_Light2")
    plt.figure(figsize=(10, 10), frameon=False)
    plot_list = [*pos_list, pos_list[0]]
    plt.plot(*zip(*plot_list))
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"div/boundary_{g}_{pts}.png")
    # plt.show()

    fig, ax = plt.subplots(figsize=(10, 10), frameon=False)
    ax.matshow(frac.T, cmap=emr, aspect="equal")
    ax.plot(*zip(*plot_list), "k", lw=3)
    # ax.grid()
    ax.autoscale(False)
    # ax.set_xlim(0,max_idx)
    # ax.set_ylim(0,max_idx)
    fig.savefig(f"classification_{g}_{pts}.png")
    # plt.show()
