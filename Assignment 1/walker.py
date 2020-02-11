# Using numbas just-in-time compilation for both functions and an entire class
from numba import jit, int8, int32, jitclass, typeof, types

tuple_ = types.containers.UniTuple(int32,2)
spec = [('position',tuple_), ('orient',int8)]

@jitclass(spec)
class Walker(object):
    
    def __init__(self,i=0,j=0):
        """
        Sets the internal variables.

        """
        self.position = (i,j)
        self.orient = 0


    def r(self):
        """
        Makes the walker turn right
    
        """
        self.orient = (self.orient + 1) % 4
    
    
    def l(self):
        """
        Makes the walker turn left
    
        """
        self.orient = (self.orient -1) % 4
    
    
    def f(self,L):
        """
        Function to propagate the walker a distance L forwards
    
        """
        if self.orient == 0: # Beveger seg mot høyre
            pos = self.position[0], self.position[1] + L
            self.position = pos
        elif self.orient == 1: # Beveger seg nedover
            pos = self.position[0] + L, self.position[1]
            self.position = pos
        elif self.orient == 2: # Beveger seg mot venstre
            pos = self.position[0], self.position[1] - L
            self.position = pos
        elif self.orient == 3: # Beveger seg oppover
            pos = self.position[0]-L, self.position[1]
            self.position = pos


@jit
def koch_walker(depth, i=0, j=0, points_between=0):
    """
    Creates a walker object and calls the recursive fractal method.
    Benchmark times:    (L, depth)
                        ( 512, 3)   332 ms ± 21.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
                        (1024, 3)   387 ms ± 41.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
                        (2048, 4)   2.59 s ± 45.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
                        (2048, 5)   21.3s
                        (8192, 5)   21.6 s ± 555 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    :Returns: A dictionary consisting of boundary points of the fractal.
    """
    walker = Walker(i,j)
    pos_dict = {}
    for _ in range(4):
        koch_recursion(walker, pos_dict, depth, points_between)
        walker.r()    
    return pos_dict


@jit
def koch_recursion(walker, pos_dict, depth, points_between = 0):
    """
    The recursion algorithm of the fractal.
    
    """
    #assert isinstance(depth,int)
    if depth ==0:
        for i in range(points_between+1):
            pos_dict[(tuple(map(int, walker.position))] = True       
            walker.f(1)
    else:
        #L /= 4.0
        koch_recursion(walker, pos_dict,depth-1, points_between) # move forward
        walker.l()
        koch_recursion(walker, pos_dict,depth-1, points_between) # move forward
        walker.r()
        koch_recursion(walker, pos_dict,depth-1, points_between) # move forward
        walker.r()
        koch_recursion(walker, pos_dict,depth-1, points_between) # move forward
        koch_recursion(walker, pos_dict,depth-1, points_between) # move forward
        walker.l()
        koch_recursion(walker, pos_dict,depth-1, points_between) # move forward
        walker.l()
        koch_recursion(walker, pos_dict,depth-1, points_between) # move forward
        walker.r()
        koch_recursion(walker, pos_dict,depth-1, points_between) # move forward