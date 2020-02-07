from walker import *

def create_lattice(g)
    """ 
    :param: g: Recursion depth (resolution)
    :returns: lattice and dictionary of boundary points.
    # vurder om punktene bør skrives til fil.

    """
    assert isinstance(g,int)
    
    N = 2*4**(g+1) # ~ca. the necessary resolution for the given depth, plus some slack.

    boundary_array = np.array(koch_walker(N,g))

    return lattice


def normalize(pointlist):
    """
    Normalize the numpy array to have minimums at index 0 
    
    """
    pointlist[:,0] += np.abs(np.min(pointlist[:,0], initial=0))
    pointlist[:,1] += np.abs(np.min(pointlist[:,1], initial=0))
    
    return

def save_grid(boundary_array, depth, N):
    """ 
    Implement filesaving

    """
    pathstr = f"boundary_grids/{N}_{depth}"
    np.savefile(pathstr, boundary_array)


def load_grid(depth, N):
    
    pathstr = f"boundary_grids/{N}_{depth}"
    try:
        arr = np.load(pathstr)
        return arr
    except IOError as e:
        print(e)
        raise e
    
        