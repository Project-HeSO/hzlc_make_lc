"""
Remove outputs of analysis of an observed day.
Pass filepath of setting.py file by the command line argument, then remove the output files.

Remove following files/directories (you can choose which files to remove by editing this file),
    * fits_header_2020....hdf5
    * light_curve_TMQ....pickle
    * detected_sources/20200401/light_curve/
    * detected_sources/20200401/movie/
    * catalog_stars_TMQ....pickle

Remove following files if the stars are only observed the given observed date. Otherwise, only the data of the observed day is cleared from the HDF5 files.
    * light_curve_....hdf5 (e.g. light_curve_Gaia-DR2_000....hdf5
    * movie_....hdf5
    * symbolic link to the above files (if only observed the given date)


Example
=======
    $ python remove_output.py /mnt/tomoe/out_analysis/log/20191107/20200818_145917/settings_20200818_145917.py

Author: Kojiro Kawana
"""

import os
import sys
import shutil
import time
import datetime
import glob
import logging
import numpy as np
import pandas as pd
import h5py

import read_output
import prepare_run
from utils import get_fits_frame_file_list

run_dry = False# do not remove files nor remove key from the HDF5 files, while the removed file list is returned
remove_fits_headers                      = True
remove_light_curve_pickle                = True
remove_detected_sources_light_curve_hdf5 = True
remove_detected_sources_movie_npy        = True
remove_catalog_pickle                    = False
remove_light_curve_hdf5                  = False
remove_movie_hdf5                        = True
remove_symbolyc_link_target_reference    = True


def is_observed_only_this_date(date_list, date):
    """
    Check whether a star is observed only on a given date.

    arguments
    =========
    date_list : list
        list of dates
    date : str
        observed date

    return
    ======
    flag : bool
    """
    return (len(date_list) == 1) and (date_list[0] == date)


def main():
    """
    Main function.

    return
    ======
    list_removed_files: list of str
        list of removed file including HDF5 files whose key, ``date_obs``, is removed while the HDF5 is not removed.
    """

    # import setting.py file given as the command line argument
    if (len(sys.argv) != 2):
        raise KeyError("Command line argument is wrong!\nExecute 'python remove_output.py PATH_TO_SETTING.PY'")
    shutil.copy(sys.argv[1], "tmp_setting.py")
    import tmp_setting
    os.remove("tmp_setting.py")

    list_removed_files = []

    run_start_time = prepare_run.get_start_time_str()
    dir_output = os.path.join(os.path.dirname(sys.argv[1]), "..")
    logger = prepare_run.prepare_logger_remove_output(dir_output, run_start_time)
    logger.info("Start\n")

    logger.info("run_dry = {:}".format(run_dry))
    logger.info("remove_fits_headers                      = {:}".format(remove_fits_headers                     ))
    logger.info("remove_light_curve_pickle                = {:}".format(remove_light_curve_pickle               ))
    logger.info("remove_catalog_pickle                    = {:}".format(remove_catalog_pickle                   ))
    logger.info("remove_light_curve_hdf5                  = {:}".format(remove_light_curve_hdf5                 ))
    logger.info("remove_movie_hdf5                        = {:}".format(remove_movie_hdf5                       ))
    logger.info("remove_detected_sources_light_curve_hdf5 = {:}".format(remove_detected_sources_light_curve_hdf5))
    logger.info("remove_detected_sources_movie_npy        = {:}".format(remove_detected_sources_movie_npy       ))
    logger.info("remove_symbolyc_link_target_reference    = {:}\n".format(remove_symbolyc_link_target_reference   ))


    if remove_fits_headers:
        fpath = os.path.join(tmp_setting.directory_fits_header_out, "fits_headers_"+str(tmp_setting.date_obs)+".hdf5")
        try:
            if not run_dry:
                os.remove(fpath)
            list_removed_files.append(fpath)
            logger.info("Finish removing fits headers")
        except Exception as e:
            logger.error(e)


    if remove_light_curve_pickle: 
        fpaths = get_fits_frame_file_list(tmp_setting.date_obs, tmp_setting.dir_output["light_curve_pickle"])
        try:
            if not run_dry:
                __ = [os.remove(fpath) for fpath in fpaths]
            list_removed_files.extend(fpaths)
            logger.info("Finish removing light curve pickle")
        except Exception as e:
            logger.error(e)

    if remove_detected_sources_light_curve_hdf5 and remove_detected_sources_movie_npy:
        fpath = os.path.join(tmp_setting.dir_output["detected_sources"], tmp_setting.date_obs)
        try:
            if not run_dry:
                shutil.rmtree(fpath)
            list_removed_files.append(fpath)
            logger.info("Finish removing detected sources")
        except Exception as e:
            logger.error(e)
    elif remove_detected_sources_light_curve_hdf5:
        fpath = os.path.join(tmp_setting.dir_output["detected_sources"], tmp_setting.date_obs, "light_curve")
        try:
            if not run_dry:
                shutil.rmtree(fpath)
            list_removed_files.append(fpath)
            logger.info("Finish removing detected sources light curve")
        except Exception as e:
            logger.error(e)
    elif remove_detected_sources_movie_npy:
        fpath = os.path.join(tmp_setting.dir_output["detected_sources"], tmp_setting.date_obs, "movie")
        try:
            if not run_dry:
                shutil.rmtree(fpath)
            list_removed_files.append(fpath)
            logger.info("Finish removing detected sources movie")
        except Exception as e:
            logger.error(e)


    if remove_light_curve_hdf5 or remove_movie_hdf5:
        fpaths = get_fits_frame_file_list(tmp_setting.date_obs, tmp_setting.dir_output["catalog_pickle"])
        dfs_catalog = []
        for fpath in fpaths:
            df_catalog = read_output.read_catalog_stars_pickle(fpath)[["catalog_name", "source_id", "is_target"]]
            dfs_catalog.append(df_catalog)
        df_catalog_all_stars = pd.concat(dfs_catalog)
        df_catalog_all_stars = df_catalog_all_stars.drop_duplicates()
        df_catalog_all_stars.reset_index(drop=True, inplace=True)
        df_catalog_all_stars[["catalog_name", "source_id"]] = df_catalog_all_stars[["catalog_name", "source_id"]].astype(str)
        logger.info("Finish reading catalog stars")
        logger.info("Nstars = {:}\n".format(df_catalog_all_stars.shape[0]))

        for i in df_catalog_all_stars.index:
            # debug
            #  if i > 10: continue
            if i % 100 == 0 or i == df_catalog_all_stars.index[-1]:
                logger.info("i_star = {:}".format(i))

            if remove_light_curve_hdf5:
                fpath = os.path.join(tmp_setting.directory_light_curve_out, "light_curve_" + df_catalog_all_stars.loc[i, "catalog_name"] + "_" + df_catalog_all_stars.loc[i, "source_id"] + ".hdf5")
                hf = h5py.File(fpath, "r+")
                keys = list(hf.keys())
                keys.remove("header")
                if not is_observed_only_this_date(keys, tmp_setting.date_obs):
                    try:
                        if not run_dry:
                            del(hf[tmp_setting.date_obs])
                        hf.close()
                        list_removed_files.append(fpath)
                    except Exception as e:
                        logger.error(e)
                        hf.close()
                else:
                    hf.close()
                    try:
                        if not run_dry:
                            os.remove(fpath)
                        list_removed_files.append(fpath)
                    except Exception as e:
                        logger.error(e)
                    if remove_symbolyc_link_target_reference and tmp_setting.make_directory_target_reference:
                        if df_catalog_all_stars.loc[i, "is_target"]:
                            fpath = os.path.join(tmp_setting.directory_light_curve_out
                                    , "target"
                                    , "light_curve_" + df_catalog_all_stars.loc[i, "catalog_name"] + "_" + df_catalog_all_stars.loc[i, "source_id"] + ".hdf5")
                        else:
                            fpath = os.path.join(tmp_setting.directory_light_curve_out
                                    , "reference"
                                    , "light_curve_" + df_catalog_all_stars.loc[i, "catalog_name"] + "_" + df_catalog_all_stars.loc[i, "source_id"] + ".hdf5")
                        try:
                            if not run_dry:
                                os.remove(fpath)
                            list_removed_files.append(fpath)
                        except Exception as e:
                            logger.error(e)

            if remove_movie_hdf5:
                fpath = os.path.join(tmp_setting.directory_movie_out, "movie_" + df_catalog_all_stars.loc[i, "catalog_name"] + "_" + df_catalog_all_stars.loc[i, "source_id"] + ".hdf5")
                hf = h5py.File(fpath, "r+")
                keys = list(hf.keys())
                keys.remove("header")
                if not is_observed_only_this_date(keys, tmp_setting.date_obs):
                    try:
                        if not run_dry:
                            del(hf[tmp_setting.date_obs])
                        hf.close()
                        list_removed_files.append(fpath)
                    except Exception as e:
                        logger.error(e)
                        hf.close()
                else:
                    hf.close()
                    try:
                        if not run_dry:
                            os.remove(fpath)
                        list_removed_files.append(fpath)
                    except Exception as e:
                        logger.error(e)
                    if remove_symbolyc_link_target_reference and tmp_setting.make_directory_target_reference:
                        if df_catalog_all_stars.loc[i, "is_target"]:
                            fpath = os.path.join(tmp_setting.directory_movie_out
                                    , "target"
                                    , "movie_" + df_catalog_all_stars.loc[i, "catalog_name"] + "_" + df_catalog_all_stars.loc[i, "source_id"] + ".hdf5")
                        else:
                            fpath = os.path.join(tmp_setting.directory_movie_out
                                    , "reference"
                                    , "movie_" + df_catalog_all_stars.loc[i, "catalog_name"] + "_" + df_catalog_all_stars.loc[i, "source_id"] + ".hdf5")
                        try:
                            if not run_dry:
                                os.remove(fpath)
                            list_removed_files.append(fpath)
                        except Exception as e:
                            logger.error(e)
        logger.info("Finish removing light curve / movie HDF5 (+ symbolic link)")


    if remove_catalog_pickle:
        fpaths = get_fits_frame_file_list(tmp_setting.date_obs, tmp_setting.dir_output["catalog_pickle"])
        try:
            if not run_dry:
                __ = [os.remove(fpath) for fpath in fpaths]
            list_removed_files.extend(fpaths)
            logger.info("Finish removing catalog pickle")
        except Exception as e:
            logger.error(e)

    # output removed files
    fpath = os.path.join(dir_output, run_start_time, "removed_files.txt")
    with open(fpath, mode="w") as f:
        f.write("\n".join(list_removed_files))

    logger.info("Finish all\n")
    return list_removed_files

if __name__ == "__main__":
    main()

