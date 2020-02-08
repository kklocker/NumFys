import numpy as np
from walker import *
import time

def create_lattice(g):
    """ 
    :param: g: Recursion depth (resolution)
    :returns: lattice and dictionary of boundary points.
    # vurder om punktene bør skrives til fil.

    """
    assert isinstance(g,int)
    
    N = 2*4**(g+1) # ~ca. the necessary resolution for the given depth, plus some slack.

    boundary_array = np.array(list(koch_walker(N,g)))
        
    return boundary_array, N


def normalize(pointlist, points_between_edges = 0):
    """
    Normalize the numpy array to have minimums at index 0 
    
    """

    delta = np.max(np.abs(pointlist[1]-pointlist[0]))
    if delta&(points_between_edges + 1) !=0:
        # Multiply value of each point by (points_between+1) to get lcm

        
    pointlist[:,0] += np.abs(np.min(pointlist[:,0], initial=0))
    pointlist[:,1] += np.abs(np.min(pointlist[:,1], initial=0))
    
    return

def save_grid(boundary_array, depth, N):
    """ 
    Implement filesaving

    """
    pathstr = f"boundary_grids/{N}_{depth}"
    np.save(pathstr, boundary_array)


def load_grid(depth, N):
    
    pathstr = f"boundary_grids/{N}_{depth}"
    try:
        arr = np.load(pathstr)
        return arr
    except IOError as e:
        print(e)
        raise e



if __name__ == "__main__":
    g = 5
    st = time.time()
    lattice, N = create_lattice(g)
    print(f"Total time elapsed: {time.time()-st} \t Time lattice creation: {time.time()-st}")
    st1 = time.time()
    normalize(lattice)
    print(f"Total time elapsed: {time.time()-st} \t Time normalize: {time.time()-st1}")
    st2 = time.time()
    save_grid(lattice,g,N)
    print(f"Total time elapsed: {time.time()-st} \t Time save: {time.time()-st2}")