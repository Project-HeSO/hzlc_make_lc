# coding: utf-8
"""
Merge files which are splitted due to MPI parallel computing (*_rank{:d}.*) to one file (*.*).

Author: Kojiro Kawana
"""
from settings_path import *
from settings_common import *
    
import os
import time
import glob
import numpy as np
import pandas as pd
import h5py
from mpi4py import MPI

import read_output
from errors_defined import *
import logging
import manage_job
import prepare_run
import utils

def remove_rank_from_file_path_str(org_file_path):
    """
    remove "_rank{:03d}" or "_proc{:03d}" from original file path

    arguments
    =========
    org_file_path : str
        original file path

    returns
    =======
    output_file_path :str
    """

    if "_rank" in org_file_path:
        split_str = "_rank"
    elif "_proc" in org_file_path:
        split_str = "_proc"

    fpath_forward, raw_fpath_backward = org_file_path.split(split_str)
    # assume that process id is given by {:03d}
    output_file_path = fpath_forward + raw_fpath_backward[3:]

    return output_file_path


def read_splitted_file_list_in_directory(directory):
    """
    Returns list of files splitted by MPI processes structured by star id.
    Search for files with their names having "_rank" or "_proc"

    arguments
    =========
    directory : str
        directory path where splitted-files are searched.

    returns
    =======
    dict_files : dict
        dictionary where items are:

            {``file_path_without_"_rank/_proc"`` : numpy.array of files having ``file_path_with_"_rank/_proc"``, 
            ...}
    """
    list_files = []
    list_files.extend(glob.glob("{:}/*_proc*".format(directory)))
    list_files.extend(glob.glob("{:}/*_rank*".format(directory)))
    list_files.sort()

    output_file_list = list(map(lambda path: remove_rank_from_file_path_str(path), list_files))

    files_array = np.array(list_files)
    output_file_array = np.array(output_file_list)

    dict_files = {}

    output_file_set  = set(output_file_list)
    for output_file in output_file_set:
        mask = output_file_array == output_file
        dict_files[output_file] = files_array[mask]

    return dict_files


def merge_fits_header_pickle_files(paths_fits_header, output_path_fits_header, remove_files_splitted=True):
    """
    Merge pickle files splitted by FRAME_IDs into 1 HDF5 file having FITS header info.

    arguments
    =========
    paths_fits_header       : list
        paths of HDF5 files conating FITS header.
    ourput_path_fits_header : str
        filepath of HDF5 file merged
    remove_files_splitted   : bool
        whether remove the unmergerd files. Defaults to True.

    Note
    ====
    Newer astropy fits.open() returns 'WCS_VALID', which consits of True or nan.
    To output the column, the format of df.to_hdf must be 'fixed' rather than 'table

    """

    df_total  = []
    for path in paths_fits_header:
        df = pd.read_pickle(path)
        df_total.append(df)
    df_total = pd.DataFrame(df_total)

    df_total.sort_values("FRAME_ID", inplace=True)
    df_total.reset_index(drop=True, inplace=True)

    #  df_total.to_hdf(output_path_fits_header, mode="a", key="data", format="table")
    df_total.to_hdf(output_path_fits_header, mode="a", key="data", format="fixed")

    if remove_files_splitted:
        for path in paths_fits_header:
            os.remove(path)

    return

def merge_fits_header_hdf5_files(paths_fits_header, output_path_fits_header, remove_files_splitted=True):
    """
    Merge HDF5 files of a certain star splitted by MPI processes into 1 HDF5 file having FITS header info.

    arguments
    =========
    paths_fits_header       : list
        paths of HDF5 files conating FITS header.
    ourput_path_fits_header : str
        filepath of HDF5 file merged
    remove_files_splitted   : bool
        whether remove the unmergerd files. Defaults to True.
    """

    df_total  = []
    for path in paths_fits_header:
        df = read_output.read_fits_header_hdf5(path)
        df_total.append(df)
    df_total = pd.concat(df_total)

    df_total.sort_values("FRAME_ID", inplace=True)
    df_total.reset_index(drop=True, inplace=True)

    df_total.to_hdf(output_path_fits_header, mode="a", key="data", format="table")

    if remove_files_splitted:
        for path in paths_fits_header:
            os.remove(path)

    return


def merge_light_curve_hdf5_files(paths_in, output_dir, date_obs, df_log, index_df_log, remove_files_splitted=True):
    """
    Merge HDF5 files of a certain star splitted by MPI processes into 2 light curve & movie HDF5 file.

    If settings.make_directory_target_reference == True, additionaly make symbolic link under target/refernce directory.
        
        e.g. ``light_curve/target/light_curve_gaia_0000000001.hdf5 -> light_curve/light_curve_gaia_0000000001.hdf5``

    arguments
    =========
    paths_in    : list
        paths of splitted-HDF5 files 
    ourput_dir  : str
        directory path where merged HDF5 file is output.
    date_obs    : str
        observed date: e.g. 20200401
    df_log  : DataFrame
        DataFrame to save time for each procedure
    index_df_log : int
        index of the current job in df_log
    remove_files_splitted : bool
        whether remove the unmergerd files. Defaults to True.


    returns
    =======
    output_path : str
        filepath of HDF5 file output

    Raises
    ======
    HDF5WriteError: Exception
        if writing HDF5 file is failed, notice the error and HDF5 file
    """

    time_read   = 0.
    time_merge  = 0.
    time_write  = 0.
    time_remove = 0.

    df_total = []
    for i, path in enumerate(paths_in):
        time_ = time.time()
        df = pd.read_hdf(path, key=date_obs)
        time_read += time.time() - time_

        time_ = time.time()
        df_total.append(df)
        time_merge += time.time() - time_

        if i==0:
            time_ = time.time()
            df_header = pd.read_hdf(path, key="header")
            time_read += time.time() - time_


    time_ = time.time()
    df_total = pd.concat(df_total)
    df_total.drop_duplicates(inplace=True)
    df_total.sort_values("utc_frame", inplace=True)
    df_total.reset_index(drop=True, inplace=True)
    time_merge += time.time() - time_

    time_ = time.time()
    output_path = os.path.join(output_dir, os.path.basename(remove_rank_from_file_path_str(paths_in[0])))
    try:
        if not os.path.exists(output_path):
            df_header.to_hdf(output_path, mode="w", key="header")
        df_total.to_hdf(output_path, mode="a", key=date_obs, format="table")

        if make_directory_target_reference:
            if df_header["is_target"]:
                path_symlink = os.path.join(os.path.dirname(output_path), "target",    os.path.basename(output_path))
            else:
                path_symlink = os.path.join(os.path.dirname(output_path), "reference", os.path.basename(output_path))
            if not os.path.exists(path_symlink):
                os.symlink(output_path, path_symlink)

    except Exception as e:
        raise HDF5WriteError("Writing {:} is fialed!".format(output_path))
    time_write = time.time() - time_

    time_ = time.time()
    if remove_files_splitted:
        for path in paths_in:
            os.remove(path)
    time_remove = time.time() - time_

    df_log.loc[index_df_log, ["time_read", "time_merge", "time_write", "time_remove"]] = time_read, time_merge, time_write, time_merge

    return output_path

def merge_movie_hdf5_files(paths_in, output_dir, date_obs, df_log, index_df_log, remove_files_splitted=True):
    """
    Merge HDF5 files of a certain star splitted by MPI processes into 2 light curve & movie HDF5 file.

    arguments
    =========
    paths_in    : list
        paths of splitted-HDF5 files 
    ourput_dir  : str
        directory path where merged HDF5 file is output.
    date_obs    : str
        observed date: e.g. 20200401
    df_log  : DataFrame
        DataFrame to save time for each procedure
    index_df_log : int
        index of the current job in df_log
    remove_files_splitted : bool
        whether remove the unmergerd files. Defaults to True.

    returns
    =======
    output_path : str
        filepath of HDF5 file output

    Raises
    ======
    HDF5WriteError: Exception
        if writing HDF5 file is failed, notice the error and HDF5 file
    """

    time_read   = 0.
    time_merge  = 0.
    time_write  = 0.
    time_remove = 0.

    utc_total                    = []
    movie_total_catalog_position = []
    movie_total_centroid         = []
    for i, path in enumerate(paths_in):
        time_ = time.time()
        __, utc, movie_catalog_position, movie_centroid = read_output.read_movie_hdf5(path, date=date_obs, read_header=False)
        time_read += time.time() - time_

        time_ = time.time()
        utc_total.append(utc[0])
        movie_total_catalog_position.append(movie_catalog_position[0])
        movie_total_centroid.append(movie_centroid[0])
        time_merge += time.time() - time_

        if i==0:
            time_ = time.time()
            df_header = pd.read_hdf(path, key="header")
            time_read += time.time() - time_


    time_ = time.time()

    utc_total = pd.concat(utc_total)
    utc_total.reset_index(drop=True, inplace=True)
    utc_total.drop_duplicates(inplace=True)
    utc_total.sort_values("utc_frame", inplace=True)
    index_sorted_by_utc = utc_total.index
    utc_total.reset_index(drop=True, inplace=True)

    movie_total_catalog_position = np.concatenate(movie_total_catalog_position)
    movie_total_catalog_position = movie_total_catalog_position[index_sorted_by_utc]
    movie_total_centroid         = np.concatenate(movie_total_centroid)
    movie_total_centroid         = movie_total_centroid[index_sorted_by_utc]

    time_merge += time.time() - time_


    time_ = time.time()
    output_path = os.path.join(output_dir, os.path.basename(remove_rank_from_file_path_str(paths_in[0])))
    try:
        if not os.path.exists(output_path):
            df_header.to_hdf(output_path, mode="w", key="header")
        with h5py.File(output_path, mode="a") as hf:
            hf.create_group(date_obs)
            hf.create_dataset(date_obs + "/movie_catalog_position", data=movie_total_catalog_position, chunks=True, maxshape=(None, None, None))
            hf.create_dataset(date_obs + "/movie_centroid",         data=movie_total_centroid,         chunks=True, maxshape=(None, None, None))
            utc_total.to_hdf(output_path, mode="a", key="{:}/utc".format(date_obs), format="table")

        if make_directory_target_reference:
            if df_header["is_target"]:
                path_symlink = os.path.join(os.path.dirname(output_path), "target",    os.path.basename(output_path))
            else:
                path_symlink = os.path.join(os.path.dirname(output_path), "reference", os.path.basename(output_path))
            if not os.path.exists(path_symlink):
                os.symlink(output_path, path_symlink)

    except Exception as e:
        raise HDF5WriteError("Writing {:} is fialed!".format(output_path))
    time_write = time.time() - time_

    time_ = time.time()
    if remove_files_splitted:
        for path in paths_in:
            os.remove(path)
    time_remove = time.time() - time_

    df_log.loc[index_df_log, ["time_read", "time_merge", "time_write", "time_remove"]] = time_read, time_merge, time_write, time_merge

    return output_path


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    run_start_time = prepare_run.get_start_time_str()
    logger = prepare_run.prepare_logger_merge(os.path.join(directory_log_merge, date_obs), run_start_time )
    logger.info("Start processing\n")

    if (rank == 0):
        for dir_ in [directory_fits_header_out, directory_light_curve_out, directory_movie_out]:
            os.makedirs(dir_, exist_ok=True)
            if (dir_ in [directory_light_curve_out, directory_movie_out]) and make_directory_target_reference:
                for tmp in ["target", "reference"]:
                    dir2_ = os.path.join(dir_, tmp)
                    os.makedirs(dir2_, exist_ok=True)

    if (rank == size-1): # in order to reduce task for rank==0 process
        time_start = time.time()
        paths_fits_header = utils.get_fits_frame_file_list(date_obs, directory_fits_header_in)
        output_path_fits_header = os.path.join(directory_fits_header_out, "fits_headers_" + str(date_obs) + ".hdf5")
        merge_fits_header_pickle_files(paths_fits_header, output_path_fits_header, remove_files_splitted)
        #  dict_files_fits_header = read_splitted_file_list_in_directory(date_obs, directory_fits_header_in)
        #  for key, fits_header_files in dict_files_fits_header.items():
        #      # only one-loop is assumed
        #      output_path_fits_header = os.path.join(directory_fits_header_out, os.path.basename(key))
        #      merge_fits_header_hdf5_files(fits_header_files, output_path_fits_header, remove_files_splitted)
        logger.info("Elapsed_time_to_merge_fits_header: {:} {:}".format(output_path_fits_header, time.time() - time_start))

    if (rank == 0):
        dict_files_lc    = read_splitted_file_list_in_directory(directory_light_curve_in)
        dict_files_movie = read_splitted_file_list_in_directory(directory_movie_in)
    else:
        dict_files_lc    = None
        dict_files_movie = None
    dict_files_lc = comm.bcast(dict_files_lc, root=0)
    dict_files_movie = comm.bcast(dict_files_movie, root=0)

    # Assume that dictionary is ordered. This is valid for Python >= 3.6
    list_paths_lc    = list(dict_files_lc.values())
    list_paths_movie = list(dict_files_movie.values())

    list_N_files = list(map(lambda x: len(x), list_paths_movie))
    array_rank_assigned = manage_job.assign_jobs_to_each_mpi_process_merge(list_N_files, size)
    if save_catalog_stars_light_curve_hdf5:
        list_paths_lc_this_rank    = np.array(list_paths_lc)[array_rank_assigned == rank]
    list_paths_movie_this_rank = np.array(list_paths_movie)[array_rank_assigned == rank]

    if save_catalog_stars_light_curve_hdf5:
        df_log = manage_job.make_df_log_merge(dict_files_lc)
        df_log = df_log.loc[array_rank_assigned == rank].reset_index(drop=True)
        for i, (key, paths_lc) in enumerate(zip(dict_files_lc.keys(), list_paths_lc_this_rank)):
            df_log.loc[i, "status_analysis"] = 1
            time_start = time.time()
            try:
                merge_light_curve_hdf5_files(paths_lc, directory_light_curve_out, date_obs, df_log, i, remove_files_splitted)
                df_log.loc[i, "time_total"] = time.time() - time_start
                df_log.loc[i, "status_analysis"] = 2
            except Exception as e:
                logger.exception("Unclassified error while processing {:}\nSkip mergeing files related to output this file\n{:}".format(key, e))
                df_log.loc[i, "time_total"] = time.time() - time_start
                df_log.loc[i, "status_analysis"] = 50
                continue

        df_log = df_log[df_log["status_analysis"] != 0]
        dfs_log = comm.gather(df_log, root=0)
        df_log_total = []
        if (rank == 0):
            for df_log in dfs_log:
                df_log_total.append(df_log)
            df_log_total = pd.concat(df_log_total)
            df_log_total = df_log_total.sort_values("file_group").reset_index(drop=True)
            df_log_total.to_csv(os.path.join(directory_log_merge, date_obs, run_start_time, "df_log_merge_lc.csv".format(rank)), mode="a", sep=" ", na_rep="NaN")

    df_log = manage_job.make_df_log_merge(dict_files_movie)
    df_log = df_log.loc[array_rank_assigned == rank].reset_index(drop=True)
    for i, (key, paths_movie) in enumerate(zip(dict_files_movie.keys(), list_paths_movie_this_rank)):
        df_log.loc[i, "status_analysis"] = 1
        time_start = time.time()
        try:
            merge_movie_hdf5_files(paths_movie, directory_movie_out, date_obs, df_log, i, remove_files_splitted)
            df_log.loc[i, "time_total"] = time.time() - time_start
            df_log.loc[i, "status_analysis"] = 2
        except Exception as e:
            logger.exception("Unclassified error while processing {:}\nSkip mergeing files related to output this file\n{:}".format(key, e))
            df_log.loc[i, "time_total"] = time.time() - time_start
            df_log.loc[i, "status_analysis"] = 50
            continue

    df_log = df_log[df_log["status_analysis"] != 0]
    dfs_log = comm.gather(df_log, root=0)
    df_log_total = []
    if (rank == 0):
        for df_log in dfs_log:
            df_log_total.append(df_log)
        df_log_total = pd.concat(df_log_total)
        df_log_total = df_log_total.sort_values("file_group").reset_index(drop=True)
        df_log_total.to_csv(os.path.join(directory_log_merge, date_obs, run_start_time, "df_log_merge_movie.csv".format(rank)), mode="a", sep=" ", na_rep="NaN")


    logger.info("Finish processing\n")
    return


if __name__ == "__main__":
    main()

