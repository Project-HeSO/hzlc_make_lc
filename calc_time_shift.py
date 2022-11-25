import numpy as np

def time_shift_sec(shape, x, y):
    ''' Calculate time-shift from the (1,1) pixel.

    Args:
        shape: ndarray contains (nx,ny)
        x: NAXIS1 coordinate of the target (zero-origin)
        y: NAXIS2 coordinate of the target (zero-origin)

    Returns:
        The timeshift value in units of second.
    '''
    nx,ny = shape
    if nx==2000 and ny==1128:
        return (930+2160+1000+np.floor(nx/4)*25)*np.floor(y/4)*100e-9
    else:
        return (930+2160+975+np.floor(nx/4)*25)*np.floor(y/4)*100e-9
