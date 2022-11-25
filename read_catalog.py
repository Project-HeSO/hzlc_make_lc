"""
Functions to read target/reference stars from catalogs

Author: Kojiro Kawana

Todo
====
* read catalog from SQL
"""

import os
import glob
import math
import pandas as pd
import numpy as np


def read_ra_dec_splitted_catalogs(fits_header, dir_catalog = "catalog", buffer_dec_deg = 2.0):
    """
    Read catalog files of Gaia and target stars covering FITS image area in a directory (dir_catalog) and return DataFrame of headers of catalogs.
    The catalog files must be splitted to 1deg x 1deg mesh.

    arguments
    =========
    fits_header : dict
        Dictionary containing information of fits header
    dir_catalog : str
        path of dierectory containing catalog of Gaia and target stars splitted by (RA, Dec).
    buffer_dec_deg : float
        We need to consider special cases where field of view covers polar point (dec = +90/-90).
        In this case, we cannot select catalog stars by rectangular filed in (RA, Dec).
        We judge this condition by the following condition:
            dec_min > (90 - buffer_dec_deg) OR dec_max < (-90 + buffer_dec_deg)


    return
    ======
    df_catalog : pandas.DataFrame
        DataFrame of catalog stars

    Todo
    ====
    * If FITS area is close to polar region, read 360 x 2 = 720 files, which makes this process severly slower (~ 2 sec). The problem is ignored now because 2 sec is small compared to total execution time per FITS file (> 30 sec), but should be solved in future.
    """

    ra_min  = np.min(fits_header["frame_radecs"][:,0])
    ra_max  = np.max(fits_header["frame_radecs"][:,0])
    dec_min = np.min(fits_header["frame_radecs"][:,1])
    dec_max = np.max(fits_header["frame_radecs"][:,1])

    if (dec_min < (90 - buffer_dec_deg) and dec_max > - (90 -buffer_dec_deg)):
        # not cover polar point
        dec_range = np.arange(math.floor(dec_min), math.ceil(dec_max))
        if (ra_max - ra_min < 300):
            # not cross ra = 360
            ra_range = np.arange(math.floor(ra_min), math.ceil(ra_max))
        else:
            ra_range1 = np.arange(0, math.ceil(ra_min))
            ra_range2 = np.arange(math.floor(ra_max), 360)
            ra_range = np.concatenate([ra_range1, ra_range2])
    else:
        # may cover polar point
        ra_range = np.arange(0, 360)
        if (dec_min > (90 - buffer_dec_deg)):
            dec_range = np.arange(math.floor(dec_min), 90)
        elif(dec_max < - (90 -buffer_dec_deg)):
            dec_range = np.arange(-90, math.ceil(dec_max))

    df_catalog = []
    for i, ra in enumerate(ra_range):
        for j, dec in enumerate(dec_range):
            filename = "Gaia-DR2_and_target_dec_{:+03d}-{:+03d}_ra_{:03d}-{:03d}_pandas.pickle".format(dec, dec+1, ra, ra+1)
            df = pd.read_pickle(os.path.join(dir_catalog, filename))
            df_catalog.append(df)
    df_catalog = pd.concat(df_catalog, ignore_index=True)

    return df_catalog


def read_all_sky_catalog_merged(dir_catalog = "catalog", file_name = "df_ohsawa_target_merged.pickle"):
    """
    Read 1 catalog of target+refernce stars covering all sky in a directory (dir_catalog) and return DataFrame of headers of catalogs.
    Read catalog at $dir_catalog/$file_name.

    arguments
    =========
    dir_catalog: str
        path of dierectory containing catalog of reference/target stars
    file_anme : str
        file_name of the catalog. The extension must be ".pickle".

    return
    ======
    df_catalog : pandas.DataFrame
        DataFrame of catalog stars
    """

    dir_catalog = os.path.abspath(dir_catalog)
    df_catalog = pd.read_pickle(os.path.join(dir_catalog, file_name))

    return df_catalog


def pickup_catalog_stars_in_frame(df_catalog, fits_header, colname_ra="ra", colname_dec="dec", buffer_dec_deg = 2.0):
    """
    Pick up stars in the FITS frame from the DataFrame of a catalog

    arguments
    =========
    df_catalog : pandas.DataFrame
        DataFrame of catalog stars
    fits_header: dict
        Dictionary of fits header
    colname_ra : str
        name of column of df_catalog expressing Ra 
    colname_dec: str
        name of column of df_catalog expressing Dec
    buffer_dec_deg : float
        We need to consider special cases where field of view covers polar point (dec = +90/-90).
        In this case, we cannot select catalog stars by rectangular filed in (RA, Dec).
        We judge this condition by the following condition:
            dec_min > (90 - buffer_dec_deg) OR dec_max < (-90 + buffer_dec_deg)

    return
    ======
    df_picked: pandas.DataFrame
        DataFrame of catalog stars in the FITS frame
    """
    wcs  = fits_header["wcs_info"]
    nax1 = fits_header["NAXIS1"]
    nax2 = fits_header["NAXIS2"]

    ra_min  = np.min(fits_header["frame_radecs"][:,0])
    ra_max  = np.max(fits_header["frame_radecs"][:,0])
    dec_min = np.min(fits_header["frame_radecs"][:,1])
    dec_max = np.max(fits_header["frame_radecs"][:,1])

    # 1st pickup stars with rectangular (Ra, Dec) field
    if (dec_min < (90 - buffer_dec_deg) and dec_max > - (90 -buffer_dec_deg)):
        # usual cases where FITS field do not cross dec = +90/-90
        df_picked = df_catalog.query("@dec_min < {:} and {:} < @dec_max".format(colname_dec, colname_dec))

        if (ra_max - ra_min < 300):
            # do not cross ra = 360
            df_picked = df_picked.query("@ra_min < {:} and {:} < @ra_max".format(colname_ra, colname_ra))
        else:
            ra_array = np.array(fits_header["frame_radecs"][:,0])
            ra_min  = np.min(ra_arary[ra_array > 180])
            ra_max  = np.max(ra_arary[ra_array < 180])
            df_picked = df_picked.query("@colname_ra > @ra_min or @colname_ra < @ra_max")
    else:
        if (dec_min > (90 - buffer_dec_deg)):
            df_picked = df_catalog.query("@dec_min < @colname_dec")
        elif(dec_max < - (90 -buffer_dec_deg)):
            df_picked = df_catalog.query("@colname_dec < @dec_max")

    # next, calc pixels of catalog stars picked up and re-pick up catalog stars
    pixes = np.array(wcs.all_world2pix(df_picked["{}".format(colname_ra)], df_picked["{}".format(colname_dec)], 0))

    # in order to avoid pandas SettingWithCopyWarning
    df_picked2 = df_picked.copy()

    df_picked2.loc[:, "x_catalog_position"] = pixes[0]
    df_picked2.loc[:, "y_catalog_position"] = pixes[1]
    df_picked2 = df_picked2.query("0 <= x_catalog_position and x_catalog_position < @nax1 and 0<= y_catalog_position and y_catalog_position < @nax2")
    df_picked2.reset_index(drop=True, inplace=True)

    return df_picked2


