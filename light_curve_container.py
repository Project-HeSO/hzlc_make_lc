"""
Utilities to handle light curve container defined as numpy structure

Author: Kojiro Kawana
"""
import copy
import datetime
import numpy as np
import pandas as pd
from calc_time_shift import time_shift_sec
from settings_common import aperture_radii_pix

_sorted_columns_light_curve_DataFrame = [
        "utc_frame",
        "time_correction_catalog_position[s]",
        "time_correction_centroid[s]",

        "flux_auto",
        "flux_auto_err",
        "pca_flux_auto",
        "pca_flux_auto_err",
        "cflux",
        "cflux_err",
        "pca_cflux",
        "pca_cflux_err",
        "flux_iso",
        "flux_iso_err",
        "pca_flux_iso",
        "pca_flux_iso_err",
        ]

for r in aperture_radii_pix:
    _sorted_columns_light_curve_DataFrame.append("flux_detected_aperture_r"+str(r))
    _sorted_columns_light_curve_DataFrame.append("flux_detected_aperture_r"+str(r)+"_err")
    _sorted_columns_light_curve_DataFrame.append("pca_flux_detected_aperture_r"+str(r))
    _sorted_columns_light_curve_DataFrame.append("pca_flux_detected_aperture_r"+str(r)+"_err")
    _sorted_columns_light_curve_DataFrame.append("flag_detected_aperture_r"+str(r))
    _sorted_columns_light_curve_DataFrame.append("flux_catalog_aperture_r"+str(r))
    _sorted_columns_light_curve_DataFrame.append("flux_catalog_aperture_r"+str(r)+"_err")
    _sorted_columns_light_curve_DataFrame.append("pca_flux_catalog_aperture_r"+str(r))
    _sorted_columns_light_curve_DataFrame.append("pca_flux_catalog_aperture_r"+str(r)+"_err")
    _sorted_columns_light_curve_DataFrame.append("flag_catalog_aperture_r"+str(r))

_sorted_columns_light_curve_DataFrame.extend([
        "x_catalog_position",
        "y_catalog_position",
        "ra_centroid",
        "dec_centroid",

        "separation_cross_match",
        "bkg_rms_global_1_pix",

        # sep returns
        "id_detected_source",
        "flag_sep",
        "flag_kron",
        "flag_flux_auto",

        "thresh",
        "npix",
        "tnpix",
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "x_centroid",
        "y_centroid",
        "x2",
        "y2",
        "xy",
        "errx2",
        "erry2",
        "errxy",
        "a",
        "b",
        "theta",
        "cxx",
        "cyy",
        "cxy",
        "cpeak",
        "peak",
        "xcpeak",
        "ycpeak",
        "xpeak",
        "ypeak",
        "kron_radius"
])


_columns_pca = [
        "pca_flux_auto",
        "pca_flux_auto_err",
        "pca_cflux",
        "pca_cflux_err",
        "pca_flux_iso",
        "pca_flux_iso_err",
        ]
for r in aperture_radii_pix:
    _columns_pca .append("pca_flux_detected_aperture_r"+str(r))
    _columns_pca .append("pca_flux_detected_aperture_r"+str(r)+"_err")
    _columns_pca .append("pca_flux_catalog_aperture_r"+str(r))
    _columns_pca .append("pca_flux_catalog_aperture_r"+str(r)+"_err")


def make_light_curve_container_of_1frame_as_DataFrame(df_catalog, fits_header, time_index):
    """
    Make pandas.DataFrame which contains fluxes, positons, etc of catalog stars in a snapshot of FITS.

    arguments
    =========
    df_catalog  : pandas.DataFrame
        DataFrame of catalog stars
    fits_header : dict
        Dictionary of fits header
    time_index  : int
        index of current time slice of FITS

    return
    ======
    df_light_curve_catalog_stars: pandas.DataFrame
        DataFrame contains fluxes of catalog stars at this frame. Shape is (N_catalog_stars, 5)
        Columns are ["time_index", "time_correction_catalog_position[s]", "x_catalog_position", "y_catalog_position", "utc_frame"].
    """

    df_light_curve_catalog_stars = df_catalog[["time_correction_catalog_position[s]", "x_catalog_position", "y_catalog_position"]].copy()
    df_light_curve_catalog_stars["utc_frame"] = fits_header["time_exp_start_arr"][time_index]
    df_light_curve_catalog_stars["time_index"] = time_index

    return df_light_curve_catalog_stars


def save_detected_sources_into_catalog_stars_DataFrame(df_light_curve_catalog_stars, df_objects_cross_matched):
    """
    Save profiles of detected sources into catalog stars light curve DataFrame cross-matched.

    arguments
    =========
    df_light_curve_catalog_stars : pandas.DataFrame
        DataFrame contains fluxes of catalog stars at this frame.
    df_objects_cross_matched     : DetectedSources (pandas.DataFrame-like)
        DataFrame of detected sources. Selected and sorted to be the same order as df_catalog.

    return
    ======
    df_light_curve_catalog_stars_updated : pandas.DataFrame
        df_light_curve_catalog_stars with columns of ``keys_copied`` are copied from df_objects_cross_matched.
    """

    keys_copied = [
        "thresh",
        "npix",
        "tnpix",
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "x_centroid",
        "y_centroid",
        "x2",
        "y2",
        "xy",
        "errx2",
        "erry2",
        "errxy",
        "a",
        "b",
        "theta",
        "cxx",
        "cyy",
        "cxy",
        "cflux",
        "cflux_err",
        "flux_iso",
        "flux_iso_err",
        "cpeak",
        "peak",
        "xcpeak",
        "ycpeak",
        "xpeak",
        "ypeak",
        "flag_sep",
        "kron_radius",
        "flag_kron",
        "flag_flux_auto",
        "flux_auto",
        "flux_auto_err",
        "ra_centroid",
        "dec_centroid",
        "time_correction_centroid[s]",
    ]
    for r in aperture_radii_pix:
        keys_copied.append("flux_detected_aperture_r"+str(r)       )
        keys_copied.append("flux_detected_aperture_r"+str(r)+"_err")
        keys_copied.append("flag_detected_aperture_r"+str(r)       ) 


    df_objects_reindexed = df_objects_cross_matched.copy()
    df_objects_reindexed.index = df_light_curve_catalog_stars.index
    df_light_curve_catalog_stars_updated = pd.concat([df_light_curve_catalog_stars, df_objects_reindexed[keys_copied]], axis=1)

    return df_light_curve_catalog_stars_updated

