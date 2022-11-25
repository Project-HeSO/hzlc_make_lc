"""
Utilities to save data on light curve and star movie

Author: Kojiro Kawana

Todo
====
* replace logger.waring() => warnings.warn()?
* remove raise Exception in output_light_curve_to_hdf5
"""
from mpi4py import MPI
import os
import numpy as np
import pandas as pd
import h5py
import utils
import pickle

import light_curve_container


def output_fits_header_to_pickle(fits_header, dir_output, logger, df_log, i_df_log):
    """
    Output pickle file containing all the information in the FITS file header.

    Names of the pickle file is:

        $(dir_output)/fits_header_$(FRAME_ID).pickle.

        e.g. ``fits_header/fits_header_TMQ12020040100000001.pickle`

    The pickle file is dict of FITS file header in key.

    arguments
    =========
    fits_header : dict
        dictionary containing all the information of fits header
    dir_output  : str
        path of directory where light curves of stars are written
    logger : logging.Logger
        Used to print log in cases where HDF5 file output are failed.
    df_log : pandas.DataFram
        DataFrame log of analysis jobs.
    i_df_log : int
        index of the log currently processed

    Raises
    ======
    PickleWriteWarning: UserWarning
        Raise warning if fail to output a pickle file, and then skip writing the HDF5 file. 

    Note
    ====
    The output pickle files are merged to a HDF5 file after processing ``merge_rank_devided_files.py``.

    Todo
    ====
    * Debug for cases where the shape of fits_headers is not unique.
        rankごとにhdf5にappendするのではなく、FRAMEごとにpickleにdumpする => merge_rank_devided_files.pyでpd.DataFrame(list_fits_dicts)でNaNを含んだDataTableにして、to_hdf5をする。
        This error occurs while handling FITS file: /alps/center/20200224/raw/rTMQ1202002240026704411.fits
        cannot match existing table structure for [EXPTIME,TELAPSE,EXPTIME1,TFRAME,DATA-FPS,EQUINOX,SANITY,F-RATIO,FOC-VAL,OBSGEO-B,OBSGEO-L,OBSGEO-H,AZIMUTH,ZD,ALTITUDE,DOMEPOS,TMP_TEL,TMP_DOME,TMP_VWS,HUM_TEL,HUM_DOME,HUM_VWS,SKYTEMP,VISRANGE,WIND_VEL,WIND_DIR,ATMOS,GAIN,RNOISE,FULLWELL,DCURRENT,Q1_TBODY,Q1_TELEC,Q1_TAMB,Q1_HUMID,Q2_TBODY,Q2_TELEC,Q2_TAMB,Q2_HUMID,Q3_TBODY,Q3_TELEC,Q3_TAMB,Q3_HUMID,Q4_TBODY,Q4_TELEC,Q4_TAMB,Q4_HUMID,VOLA_7VA,CURA_7VA,VOLA_5VA,CURA_5VA,VOLA_7VB,CURA_7VB,VOLA_5VB,CURA_5VB,VOLD_5V,CURD_5V,FMTVER,VERSION,CRPIX1,CRPIX2,CRPIX3,CRVAL1,CRVAL2,CRVAL3,CD1_1,CD1_2,CD1_3,CD2_1,CD2_2,CD2_3,CD3_1,CD3_2,CD3_3,LONPOLE,LATPOLE,A_0_2,A_1_1,A_2_0,AP_0_0,AP_0_1,AP_0_2,AP_1_0,AP_1_1,AP_2_0,B_0_2,B_1_1,B_2_0,BP_0_0,BP_0_1,BP_0_2,BP_1_0,BP_1_1,BP_2_0,ra_x0y0,ra_x1y1,ra_x0y1,ra_x1y0,dec_x0y0,dec_x1y1,dec_x0y1,dec_x1y0] on appending data
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    keys_popped = [
            "wcs_info",
            "time_exp_start",
            "time_exp_start_arr",
            "frame_pixes",
            "frame_radecs"
            ]
    fits_header["ra_x0y0"],  fits_header["ra_x1y1"],  fits_header["ra_x0y1"],  fits_header["ra_x1y0"]  = fits_header["frame_radecs"][:,0]
    fits_header["dec_x0y0"], fits_header["dec_x1y1"], fits_header["dec_x0y1"], fits_header["dec_x1y0"] = fits_header["frame_radecs"][:,1]
    for key in keys_popped:
        fits_header.pop(key)
    dict_dtypes = {k: type(v) for k, v in fits_header.items()}

    date = fits_header["DATE-OBS"].replace("-", "")
    path_output = "{:}/fits_header_{:}.pickle".format(dir_output, fits_header["FRAME_ID"])

    try:
        with open(path_output, "wb") as f:
            pickle.dump(fits_header, f)
        #  pd_series = pd.Series(fits_header)
        #  pd_series.to_frame().T.astype(dict_dtypes).to_hdf(path_output, mode="a", key="data", format="table", append=True)
    except Exception as e:
        logger.warning("PickleWriteWarning! writing {:} is failed! Skip this file\nThis error occurs while handling FITS file: {:}\n{:}\n".format(path_output, df_log.loc[i_df_log, "filepath"], e))
        df_log.loc[i_df_log, "status_analysis"] = 11
        return

    return


def output_light_curve_to_hdf5(df_catalog, df_light_curve_catalog_stars, dir_output, fits_header, logger, df_log, i_df_log):
    """
    Output HDF5 files containing light curve of each star.

    Names of the HDF5 files are:

        $(dir_output)/light_curve_$(df_catalog["catalog_name") + "_" + $(df_catalog["source_id"]) + "_proc" + $(rank) + ".hdf5"

        e.g. ``light_curve/light_curve_gaia_0000000001_proc001.hdf5``

    A HDF5 file has the following structure:
    
        * ``/``

            * ``/header`` : contains "catalog_name", "source_id", "catalog_mag", "is_target", "ra", "dec"
            * ``$(DATE-OBS)`` e.g. 20200401 : data table contains light curve info.
            * ... (e.g. 20200402 ...)

    arguments
    =========
    df_catalog: DataFrame
        DataFrame contains header information of catalog stars.
    df_light_curve_catalog_stars : DataFrame
        DataFrame contains light_curve of catalog stars. UTC information are written to HDF5.
    dir_output : str
        path of directory where light curves of stars are written
    fits_header : dict
        Dictionary containing information of fits header
    logger : logging.Logger
        Used to print log in cases where HDF5 file output are failed.
    df_log : pandas.DataFram
        DataFrame log of analysis jobs.
    i_df_log : int
        index of the log currently processed

    Raises
    ========
    HDF5WriteWarning: UserWarning
        Raise warning if fail to output a HDF5 file, and then skip writing the HDF5 file. 
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    columns_header = [
        "catalog_name", 
        "source_id", 
        "catalog_mag",
        "is_target",
        "ra",
        "dec",
    ]

    df_header = df_catalog[columns_header]

    N_stars = df_catalog.shape[0]
    for i_star in range(N_stars):
        path_output = "{:}/light_curve_{:}_{:}_proc{:03d}.hdf5".format(dir_output, df_header.loc[i_star, "catalog_name"], df_header.loc[i_star, "source_id"], rank)
        try:
            if not (os.path.exists(path_output)):
                df_header_this_star = df_header.loc[i_star]
                df_header_this_star.name = None
                df_header_this_star.to_hdf(path_output, mode="w", key="/header")

            df_light_curve_this_star = df_light_curve_catalog_stars.loc[i_star]
            df_light_curve_this_star.to_hdf(path_output, mode="a", key=fits_header["DATE-OBS"].replace("-", ""), format="table", append=True)

        except Exception as e:
            raise Exception(e)
            logger.warning("HDF5WriteWarning! writing {:} is failed! Skip this file\nThis error occurs while handling FITS file: {:}\n{:}\n".format(path_output, df_log.loc[i_df_log, "filepath"], e))
            df_log.loc[i_df_log, "status_analysis"] = 11
            continue

    return


def output_movie_to_hdf5(movie_catalog_stars, df_catalog, df_light_curve_catalog_stars, dir_output, fits_header, logger, df_log, i_df_log):
    """
    Output HDF5 files containing cutout movie of each star.

    Names of the HDF5 files are:

        $(dir_output)/movie_$(df_catalog["catalog_name") + "_" + $(df_catalog["source_id"]) + "_proc" + $(rank) + ".hdf5"

        e.g. ``movie/movie_gaia_0000000001_proc001.hdf5``

    A HDF5 file has the following structure:
    
        * ``/``

            * ``/header`` : contains "catalog_name", "source_id", "catalog_mag", "is_target", "ra", "dec"
            * ``$(DATE-OBS)`` e.g. 20200401 : data table contains light curve info.

                * ``/utc`` : pandas.DataFrame containing frame time in UTC and time correction with respect to catalog position (center of cutout movie)
                * ``/movie_catalog_position`` : cubic images (movie) contains ``movie_catalog_position``
                * ``/movie_centroid`` : cubic images (movie) contains ``movie_centroid``

            * ... (e.g. 20200402 ...)

    arguments
    =========
    movie_catalog_stars : dict of numpy array(dtype=float32)
        Keys are ["catalog_position", "detected_position"]. Shape must be (N_stars, fits_header["NAXIS3"], width_cutout, width_cutout)
    df_catalog: DataFrame
        DataFrame contains header information of catalog stars.
    df_light_curve_catalog_stars : DataFrame
        DataFrame contains light_curve of catalog stars. UTC information are written to HDF5.
    dir_output : str
        path of directory where movies of stars are written
    fits_header : dict
        Dictionary containing information of fits header
    logger : logging.Logger
        Used to print log in cases where HDF5 file output are failed.
    df_log : pandas.DataFram
        DataFrame log of analysis jobs.
    i_df_log : int
        index of the log currently processed

    Raises
    ========
    HDF5WriteWarning: UserWarning
        Raise warning if fail to output a HDF5 file, and then skip writing the HDF5 file. 
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    nax3 = fits_header["NAXIS3"]
    date_obs = fits_header["DATE-OBS"].replace("-", "")

    columns_header = [
        "catalog_name", 
        "source_id", 
        "catalog_mag",
        "is_target",
        "ra",
        "dec",
    ]

    columns_utc = [
       "utc_frame",
       "time_correction_catalog_position[s]",
       "time_correction_centroid[s]",
    ]


    df_header = df_catalog[columns_header]
    df_utc    = df_light_curve_catalog_stars[columns_utc]

    N_stars = df_catalog.shape[0]
    for i_star in range(N_stars):
        path_output = "{:}/movie_{:}_{:}_proc{:03d}.hdf5".format(dir_output, df_header.loc[i_star, "catalog_name"], df_header.loc[i_star, "source_id"], rank)

        df_utc_this_star = df_utc.loc[i_star]

        try:
            if not (os.path.exists(path_output)):
                df_header_this_star = df_header.loc[i_star]
                df_header_this_star.name = None
                df_header_this_star.to_hdf(path_output, mode="w", key="/header")

                with h5py.File(path_output, mode="a") as hf:
                    hf.create_group(date_obs)
                    hf.create_dataset("{:}/movie_catalog_position".format(date_obs), data=movie_catalog_stars["catalog_position" ][i_star], chunks=True, maxshape=(None, None, None))
                    hf.create_dataset("{:}/movie_centroid".format(        date_obs), data=movie_catalog_stars["detected_position"][i_star], chunks=True, maxshape=(None, None, None))

            else:
                with h5py.File(path_output, mode="a") as hf:
                    hf["{:}/movie_catalog_position".format(date_obs)].resize((hf["{:}/movie_catalog_position".format(date_obs)].shape[0] + nax3), axis=0)
                    hf["{:}/movie_centroid".format(date_obs)        ].resize((hf["{:}/movie_centroid".format(date_obs)        ].shape[0] + nax3), axis=0)
                    hf["{:}/movie_catalog_position".format(date_obs)][-nax3:] = movie_catalog_stars["catalog_position" ][i_star]
                    hf["{:}/movie_centroid".format(date_obs)        ][-nax3:] = movie_catalog_stars["detected_position"][i_star]

            df_utc_this_star.to_hdf(path_output, mode="a", key="{:}/utc".format(date_obs), format="table", append=True)

        except Exception as e:
            logger.warning("HDF5WriteWarning! writing {:} is failed! Skip this file\nThis error occurs while handling FITS file: {:}\n{:}\n".format(path_output, df_log.loc[i_df_log, "filepath"], e))
            df_log.loc[i_df_log, "status_analysis"] = 11
            continue

    return
