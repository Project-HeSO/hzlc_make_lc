"""
Functions to read fits file.

Author: Kojiro Kawana
"""

import warnings
import datetime
import numpy as np
import astropy.io.fits as fits
from astropy.wcs import WCS

from errors_defined import *

def read_fits_header_object(fits_file_path, **params_fits_open):
    """
    Read one fits file and return header information.
    Original keys in FITS header have capital charecters, whereas additional keys have small characters.

    arguments
    =========
    fits_file_path: str
        file path of a FITS file
    calc_WCS : bool
        If True, calculate WCS and (RA, Dec) at the edges of the frame, which takes longer time to execute (~ 40 ms if True, ~ 10 ms if False).
        Default is False.
    **params_fits_open: dict
        keyword arguments for the options of fits.open()

    return
    ======
    hdul       : HDU List
        return of fits.open()
    fits_header: dict
        Dictionary containing information of fits header + additional info
    """
    fits_header = {}

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="non-ASCII characters are present in the FITS file header and have been replaced by")
        try:
            hdul = fits.open(fits_file_path, **params_fits_open)
            raw_fits_header = hdul[0].header
            fits_header = dict(hdul[0].header)
            object_now = fits_header["OBJECT"]
            
            return object_now
        except Exception as e:
            raise FitsReadError(e)


            
def read_fits_header(fits_file_path, calc_WCS=False, **params_fits_open):
    """
    Read one fits file and return header information.
    Original keys in FITS header have capital charecters, whereas additional keys have small characters.

    arguments
    =========
    fits_file_path: str
        file path of a FITS file
    calc_WCS : bool
        If True, calculate WCS and (RA, Dec) at the edges of the frame, which takes longer time to execute (~ 40 ms if True, ~ 10 ms if False).
        Default is False.
    **params_fits_open: dict
        keyword arguments for the options of fits.open()

    return
    ======
    hdul       : HDU List
        return of fits.open()
    fits_header: dict
        Dictionary containing information of fits header + additional info
    """
    fits_header = {}

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="non-ASCII characters are present in the FITS file header and have been replaced by")
        try:
            hdul = fits.open(fits_file_path, **params_fits_open)
            raw_fits_header = hdul[0].header
            fits_header = dict(hdul[0].header)
            fits_header.pop("COMMENT") # fits_header["COMMENT"] == ""
            fits_header["HISTORY"] = str(fits_header["HISTORY"])

            fits_header["time_exp_start"]     = datetime.datetime.strptime(fits_header["UTC"], '%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc)
            fits_header["time_exp_start_arr"] = (fits_header["time_exp_start"]+ np.arange(fits_header["NAXIS3"]) * datetime.timedelta(seconds=fits_header["TFRAME"])).astype(np.dtype("datetime64[us]"))
            fits_header["frame_pixes"]        = np.array([[0,0], [fits_header["NAXIS1"],fits_header["NAXIS2"]], [0,fits_header["NAXIS2"]], [fits_header["NAXIS1"],0]])
            if calc_WCS:
                fits_header["wcs_info"]           = WCS(raw_fits_header, naxis=2)
                fits_header["frame_radecs"]       = fits_header["wcs_info"].all_pix2world(fits_header["frame_pixes"], 0)

            return hdul, fits_header
        except Exception as e:
            raise FitsReadError(e)

