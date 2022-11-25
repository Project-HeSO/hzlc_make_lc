"""
Cut target objects from FITS and save the data

Author: Kojiro Kawana
"""

import numpy as np
from astropy.nddata.utils import Cutout2D

def cutout_sources_from_FITS(fits_snapshot, xy_arr, width_cutout):
    """
    Cutout images of sources from a FITS snapshot.
    This function wraps astropy.nddata.utils.Cutout2D.

    arguments
    =========
    fits_snapshot   : numpy.array of float
        2D FITS data. same as hdul[0].data[i] - sep.Background
    xy_arr: numpy array
        x, y coordinates of centers of the cutout images. Shape must be [N_sources, 2].
        The argument should be numpy array rather than DataFrame for fast computtation
    width_cutout    : int
        width of cutout image. => shape (width_cutout, width_cutout)

    return
    ======
    images_detected_stars: numpy array of float32
        Shape: (xy_arr.shape[0], width_cutout, width_cutout).
    """

    N_sources = xy_arr.shape[0]
    images_detected_stars = np.zeros((N_sources, width_cutout, width_cutout), dtype=np.float32)

    for i in range(N_sources):
        images_detected_stars[i] = Cutout2D(fits_snapshot, xy_arr[i], width_cutout, mode="partial").data

    return images_detected_stars

