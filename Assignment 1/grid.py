import numpy as np
from walker import *
import time
import os


def create_lattice(g, points_between=0):
    """ 
    :param: g: Recursion depth (resolution)
    :returns: lattice and dictionary of boundary points.
    # vurder om punktene bør skrives til fil.

    """
    assert isinstance(g, int)

    # N = 2*4**(g+1) # ~ca. the necessary resolution for the given depth, plus some slack.

    boundary_array = np.array(list(koch_walker(g, points_between=points_between)))

    return boundary_array


def normalize(pointlist, parallell=True):
    """
    Normalize the numpy array to have minimums at index 0 and index the same as latticespacing
    returns largest index of the grid.
    """
    if parallell:
        pointlist = pointlist.copy()
    delta = np.max(np.abs(pointlist[1] - pointlist[0]))
    tmp = pointlist[:, 0] / delta
    tmp2 = pointlist[:, 1] / delta
    pointlist[:, 0] = tmp
    pointlist[:, 1] = tmp2
    pointlist[:, 0] += np.abs(np.min(pointlist[:, 0], initial=0))
    pointlist[:, 1] += np.abs(np.min(pointlist[:, 1], initial=0))
    max_idx = np.max(pointlist)
    if parallell:
        return max_idx, pointlist
    return max_idx


def get_grid(depth, points_between):

    path = f"boundary_grids/{depth}_{points_between}.npy"
    if os.path.exists(path):
        temp_dict = load_grid(path)
    else:
        temp_dict = koch_walker(depth=depth, points_between=points_between)
        save_grid(temp_dict, depth, points_between)

    return np.array(list(temp_dict))


def convert_to_dict(pointlist):
    """Converts (normalized) point list into dictionary.
    """
    pos_dict = {tuple(p): True for p in pointlist}
    return pos_dict


def save_grid(boundary_array, depth, points_between):
    """ 
    Implement filesaving

    """
    pathstr = f"boundary_grids/{depth}_{points_between}"
    np.save(pathstr, boundary_array)


def load_grid(path):
    try:
        arr = np.load(path, allow_pickle=True)
        return arr.item()
    except IOError as e:
        print(e)
        raise e


if __name__ == "__main__":
    g = 3
    points_between = 2
    st = time.time()
    lattice = create_lattice(g, points_between)
    print(
        f"Total time elapsed: {time.time()-st} \t Time lattice creation: {time.time()-st}"
    )
    st1 = time.time()
    max_idx = normalize(lattice)
    print(f"Total time elapsed: {time.time()-st} \t Time normalize: {time.time()-st1}")
    st2 = time.time()
    save_grid(lattice, g, points_between)
    print(f"Total time elapsed: {time.time()-st} \t Time save: {time.time()-st2}")
