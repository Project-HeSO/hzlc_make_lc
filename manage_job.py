"""
manage tasks analysing fits file for each MPI process

Author: Kojiro Kawana
"""

from mpi4py import MPI
import os
import glob
import numpy as np
import pandas as pd
from settings_common import aperture_radii_pix

import read_fits 


def select_fits_for_target_names(fit_paths, object_names=None):
    
    if object_names is None:
        return fit_paths
    
    fits_paths_selected  = []
    for fit_path in fit_paths: 
        object_now = read_fits.read_fits_header_object(fit_path)
        if object_now in object_names:
            fits_paths_selected.append(fit_path)
    return fits_paths_selected


def make_df_log_from_fits_directory(directories_fits_files):
    """
    Read directory(s) containing FITS files to be processed and return list of DataFrame of logs of processing times/status.
    This function wraps `make_df_log_from_fits_directory_single_dir`.

    arguments
    =========
    directories_fits_files :str or list of str
        directories containing FITS files.

    return
    ======
    list_df_log :list of pandas.DataFrame
        A DataFrame records log of the analysis-tasked.

        len(list_df_log) == len(list(``directories_fits_files``))

        Columns in the DataFrame are as follows:

            * filepath       : filepath of a fits file
            * status_analysis: status of the analysis-task

                * 0  not processd yet
                * 1  under processing
                * 2  process finished
                * 10 FitsReadError
                * 11 HDF5WriteError
                * 12 SourceDetectionError
                * 50 Unclassified error
            * t_*: time elapsed to do *
    """
    if type(directories_fits_files) is str:
        directories_fits_files_list = [directories_fits_files]
    else:
        directories_fits_files_list = directories_fits_files

    list_df_log = []
    for dir_ in directories_fits_files_list:
        list_df_log.append(make_df_log_from_fits_directory_single_dir(dir_))

    return list_df_log

def make_df_log_from_fits_directory_debug(directories_fits_files, number_fits = -1, object_names=None):
    """
    Read directory(s) containing FITS files to be processed and return list of DataFrame of logs of processing times/status.
    This function wraps `make_df_log_from_fits_directory_single_dir`.

    arguments
    =========
    directories_fits_files :str or list of str
        directories containing FITS files.

    return
    ======
    list_df_log :list of pandas.DataFrame
        A DataFrame records log of the analysis-tasked.

        len(list_df_log) == len(list(``directories_fits_files``))

        Columns in the DataFrame are as follows:

            * filepath       : filepath of a fits file
            * status_analysis: status of the analysis-task

                * 0  not processd yet
                * 1  under processing
                * 2  process finished
                * 10 FitsReadError
                * 11 HDF5WriteError
                * 12 SourceDetectionError
                * 50 Unclassified error
            * t_*: time elapsed to do *
    """
    if type(directories_fits_files) is str:
        directories_fits_files_list = [directories_fits_files]
    else:
        directories_fits_files_list = directories_fits_files

    list_df_log = []
    for dir_ in directories_fits_files_list:
        list_df_log.append(make_df_log_from_fits_directory_single_dir_debug(dir_, number_fits, object_names))

    return list_df_log



def make_df_log_from_fits_directory_single_dir_debug(dir_fits_files, number_fits = -1, object_names=None):
    """
    Read one directory containing fits files to be processed and return DataFrame of logs of processing times/status.

    arguments
    =========
    dir_fits_files: path of directory containing fits files to be processed

    return
    ======
    df_log : pandas.DataFrame
        DataFrame records log of the analysis-tasked.

        Columns in the DataFrame are as follows:

            * filepath       : filepath of a fits file
            * status_analysis: status of the analysis-task

                * 0  not processd yet
                * 1  under processing
                * 2  process finished
                * 10 FitsReadError
                * 11 HDF5WriteError
                * 12 SourceDetectionError
                * 50 Unclassified error
            * t_*: time elapsed to do *
    """

    dir_fits_files = os.path.abspath(dir_fits_files)
    fits_file_paths = glob.glob('{:}/*.fits'.format(dir_fits_files))
    fits_file_paths = fits_file_paths[:number_fits]
    fits_file_paths = select_fits_for_target_names(fits_file_paths, object_names)
    fits_file_paths.sort()

    columns = ["filepath", "status_analysis", "is_wcs_processed", "N_catalog_stars", "N_detected_sources_in_all_frames"]
    columns_time = ["t_total", "t_open_fits", "t_read_catalog", "t_pick_up_catalog_stars", "t_prepare_light_curve_container",
                    "t_read_snapshot", "t_bkg_subtraction", "t_cutout_FITS_catalog_sources", "t_detection", "t_cutout_FITS_detected_sources",
                    "t_calc_FLUX_AUTO_error", "t_time_shift_centroid", "t_kron_photometry_detected"]
    columns_time.extend(["t_catalog_aperture_photometry_r"  + str(r) for r in aperture_radii_pix])
    columns_time.extend(["t_detected_aperture_photometry_r" + str(r) for r in aperture_radii_pix])
    columns_time.extend(["t_calc_ra_dec_detected", "t_cross_match", "t_save_to_catalog_sources", "t_reshape_catalog_sources_pandas"])
    columns_time_output = ["t_output_total",
                    "t_output_light_curve", "t_output_movie",
                    "t_output_catalog_pickle",
                    "t_output_light_curve_pickle",
                    "t_output_detected_sources_light_curve", "t_output_detected_sources_movie",
                    "t_output_fits_header"]
    columns_time.extend(columns_time_output)
    columns.extend(columns_time)

    df_log = pd.DataFrame(np.zeros((len(fits_file_paths), len(columns)), dtype=np.float), columns=columns, dtype = np.float)
    df_log["{}".format(columns[0])] = fits_file_paths
    columns_int = ["status_analysis", "N_catalog_stars", "N_detected_sources_in_all_frames"]
    df_log[columns_int] = df_log[columns_int].astype(np.int32)
    df_log["{}".format(columns[2])] = df_log["{}".format(columns[2])].astype(bool)

    return df_log




def make_df_log_from_fits_directory_single_dir(dir_fits_files):
    """
    Read one directory containing fits files to be processed and return DataFrame of logs of processing times/status.

    arguments
    =========
    dir_fits_files: path of directory containing fits files to be processed

    return
    ======
    df_log : pandas.DataFrame
        DataFrame records log of the analysis-tasked.

        Columns in the DataFrame are as follows:

            * filepath       : filepath of a fits file
            * status_analysis: status of the analysis-task

                * 0  not processd yet
                * 1  under processing
                * 2  process finished
                * 10 FitsReadError
                * 11 HDF5WriteError
                * 12 SourceDetectionError
                * 50 Unclassified error
            * t_*: time elapsed to do *
    """

    dir_fits_files = os.path.abspath(dir_fits_files)
    fits_file_paths = glob.glob('{:}/*.fits'.format(dir_fits_files))
    fits_file_paths.sort()

    columns = ["filepath", "status_analysis", "is_wcs_processed", "N_catalog_stars", "N_detected_sources_in_all_frames"]
    columns_time = ["t_total", "t_open_fits", "t_read_catalog", "t_pick_up_catalog_stars", "t_prepare_light_curve_container",
                    "t_read_snapshot", "t_bkg_subtraction", "t_cutout_FITS_catalog_sources", "t_detection", "t_cutout_FITS_detected_sources",
                    "t_calc_FLUX_AUTO_error", "t_time_shift_centroid", "t_kron_photometry_detected"]
    columns_time.extend(["t_catalog_aperture_photometry_r"  + str(r) for r in aperture_radii_pix])
    columns_time.extend(["t_detected_aperture_photometry_r" + str(r) for r in aperture_radii_pix])
    columns_time.extend(["t_calc_ra_dec_detected", "t_cross_match", "t_save_to_catalog_sources", "t_reshape_catalog_sources_pandas"])
    columns_time_output = ["t_output_total",
                    "t_output_light_curve", "t_output_movie",
                    "t_output_catalog_pickle",
                    "t_output_light_curve_pickle",
                    "t_output_detected_sources_light_curve", "t_output_detected_sources_movie",
                    "t_output_fits_header"]
    columns_time.extend(columns_time_output)
    columns.extend(columns_time)

    df_log = pd.DataFrame(np.zeros((len(fits_file_paths), len(columns)), dtype=np.float), columns=columns, dtype = np.float)
    df_log["{}".format(columns[0])] = fits_file_paths
    columns_int = ["status_analysis", "N_catalog_stars", "N_detected_sources_in_all_frames"]
    df_log[columns_int] = df_log[columns_int].astype(np.int32)
    df_log["{}".format(columns[2])] = df_log["{}".format(columns[2])].astype(bool)

    return df_log



def make_df_log_from_fits_directory_single_dir(dir_fits_files):
    """
    Read one directory containing fits files to be processed and return DataFrame of logs of processing times/status.

    arguments
    =========
    dir_fits_files: path of directory containing fits files to be processed

    return
    ======
    df_log : pandas.DataFrame
        DataFrame records log of the analysis-tasked.

        Columns in the DataFrame are as follows:

            * filepath       : filepath of a fits file
            * status_analysis: status of the analysis-task

                * 0  not processd yet
                * 1  under processing
                * 2  process finished
                * 10 FitsReadError
                * 11 HDF5WriteError
                * 12 SourceDetectionError
                * 50 Unclassified error
            * t_*: time elapsed to do *
    """

    dir_fits_files = os.path.abspath(dir_fits_files)
    fits_file_paths = glob.glob('{:}/*.fits'.format(dir_fits_files))
    fits_file_paths.sort()

    columns = ["filepath", "status_analysis", "is_wcs_processed", "N_catalog_stars", "N_detected_sources_in_all_frames"]
    columns_time = ["t_total", "t_open_fits", "t_read_catalog", "t_pick_up_catalog_stars", "t_prepare_light_curve_container",
                    "t_read_snapshot", "t_bkg_subtraction", "t_cutout_FITS_catalog_sources", "t_detection", "t_cutout_FITS_detected_sources",
                    "t_calc_FLUX_AUTO_error", "t_time_shift_centroid", "t_kron_photometry_detected"]
    columns_time.extend(["t_catalog_aperture_photometry_r"  + str(r) for r in aperture_radii_pix])
    columns_time.extend(["t_detected_aperture_photometry_r" + str(r) for r in aperture_radii_pix])
    columns_time.extend(["t_calc_ra_dec_detected", "t_cross_match", "t_save_to_catalog_sources", "t_reshape_catalog_sources_pandas"])
    columns_time_output = ["t_output_total",
                    "t_output_light_curve", "t_output_movie",
                    "t_output_catalog_pickle",
                    "t_output_light_curve_pickle",
                    "t_output_detected_sources_light_curve", "t_output_detected_sources_movie",
                    "t_output_fits_header"]
    columns_time.extend(columns_time_output)
    columns.extend(columns_time)

    df_log = pd.DataFrame(np.zeros((len(fits_file_paths), len(columns)), dtype=np.float), columns=columns, dtype = np.float)
    df_log["{}".format(columns[0])] = fits_file_paths
    columns_int = ["status_analysis", "N_catalog_stars", "N_detected_sources_in_all_frames"]
    df_log[columns_int] = df_log[columns_int].astype(np.int32)
    df_log["{}".format(columns[2])] = df_log["{}".format(columns[2])].astype(bool)

    return df_log


def check_previous_job_and_exclude_finshed_log(path_previous_log, df_log_from_directory, output_file_path):
    """
    Check log file of the previous log and put flags on previously processed fits files from the current log

    arguments
    =========
    path_previous_log    : str
        file path of log file of the previous job
    df_log_from_directory: pandas.DataFrame
        DataFrame of log/job made from fits files in fits directory
    output_file_path     : str
        file path where the log file are written

    return
    ======
    df_log_total: pandas.DataFrame
        DataFrame of log/job where previouly processed log are flagged
    """
    comm = MPI.COMM_WORLD

    df_log_previous = pd.read_csv(path_previous_log, delim_whitespace=True, index_col=0)
    df_log_total = df_log_from_directory.append(df_log_previous)
    df_log_total = df_log_total.drop_duplicates("filepath", keep="last").sort_values("filepath").reset_index(drop=True)

    # output finished log to output_file_path
    if (comm.Get_rank() == 0):
        df_log_total[df_log_total["status_analysis"] > 0].reindex().to_csv(output_file_path, mode="a", sep=" ")
    return df_log_total.query("status_analysis != 2").reindex()


def assign_jobs_to_each_mpi_process(list_df_log):
    """
    Assign indexes of df_log to each mpi process

    arguments
    =========
    list_df_log : list of pandas.DataFrame
        DataFrame showing fits files, status of analysis, and times taken to be processed

    return
    ======
    df_log_merged : pandas.DataFrame
        Merged DataFrame (reindexed). The order is from the largest df.shape[0] to the smallest. 
    indexes_my_rank : numpy.array
        Indexes of the above DataFrame processed by "this" MPI porcess (the return is changed by MPI processes)

    Note
    ====
    Now we crudely use for-loop but it is not too slow. e.g. If sizes of list_df_log are ~3e4, this takes 50 ms on laptop.
    """
    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    Nproc = comm.Get_size()
    
    sizes_df_log_raw = [df.shape[0] for df in list_df_log]
    list_df_log_sorted_from_largest = [list_df_log[order] for order in np.argsort(sizes_df_log_raw)[::-1]]
    sizes_df_log_sorted_from_largest = np.sort(sizes_df_log_raw)[::-1]

    max_size_df_log = sizes_df_log_sorted_from_largest[0]
    rank_id_matrix = -1 * np.ones((max_size_df_log, len(list_df_log)), dtype=int)
    do_asign_rank_id = np.ones_like(rank_id_matrix, dtype=bool)

    for i, size_df_log in enumerate(sizes_df_log_sorted_from_largest):
        do_asign_rank_id[size_df_log:, i] = False

    N, M = rank_id_matrix.shape
    rank_id = -1
    for i in range(N):
        for j in range(M):
            if do_asign_rank_id[i,j]:
                rank_id = (rank_id+1) % Nproc
                rank_id_matrix[i, j] = rank_id

    rank_id_array = rank_id_matrix.flatten(order="F")
    rank_id_array = rank_id_array[rank_id_array >= 0]
    indexes_my_rank = np.arange(rank_id_array.size)[rank_id_array == rank]

    # merge list of df_logs to one DataFrame
    df_log_merged = pd.concat(list_df_log_sorted_from_largest, ignore_index=True)

    return df_log_merged, indexes_my_rank



def assign_jobs_to_each_mpi_process_single_dir(df_log):
    """
    Assign indexes of df_log to each mpi process

    arguments
    =========
    df_log: pandas.DataFrame
        DataFrame showing fits files, status of analysis, and times taken to be processed

    return
    ======
    i_df_log_st: int
        starting index of df_log of each_process
    i_df_log_ed: int
        ending index of df_log of each_process
    """
    comm = MPI.COMM_WORLD

    i_df_log_st_tot = 0
    i_df_log_ed_tot = df_log.shape[0]
    i_df_log_st = np.int(comm.Get_rank() * (i_df_log_ed_tot - i_df_log_st_tot) / comm.Get_size())
    i_df_log_ed = np.int((1 + comm.Get_rank()) * (i_df_log_ed_tot - i_df_log_st_tot) / comm.Get_size())

    return i_df_log_st, i_df_log_ed


###########################################
### For merge_rank_devided_files.py #######
##########################################

def assign_jobs_to_each_mpi_process_merge(task_weights, Nproc):
    """
    Equally devide tasks with given weights of the task by N processes.

    arguments
    =========
    task_weights : list/numpy.array
        1D List of weights of task
    Nproc : int
        number of processes deviding the task

    return
    ======
    array_rank_assign : numpy.array(int)
        1D array with the same size asa task_weights. Showing which rank (range(Nproc)) is assigned for the task
    """

    array_cumsum_weights = np.cumsum(task_weights)
    threshold = array_cumsum_weights[-1] / Nproc
    array_rank_assigned = (array_cumsum_weights / threshold).astype(int)
    array_rank_assigned[array_rank_assigned == Nproc] = Nproc-1

    return array_rank_assigned

def make_df_log_merge(dict_files):
    """
    Read one directory containing fits files to be processed and return DataFrame of logs of processing times/status.

    arguments
    =========
    dict_files: dict
        Path of directory containing fits files to be processed

    return
    ======
    df_log: pandas.DataFrame
        DataFrame showing fits files, status of analysis, and times taken to be processed

        Columns in the DataFrame are as follows:

            * file_group     : filepath of a fits file
            * filepaths_in   : filepath of a fits file
            * filepath       : filepath of a fits file
            * status_analysis: status of the analysis-task

                * 0  not processd yet
                * 1  under processing
                * 2  process finished
                * 50 Unclassified error
    """

    df_log = pd.DataFrame(dict_files.keys(), columns=["file_group"])
    df_log["N_files"] = [value.size for value in dict_files.values()]
    df_log["status_analysis"] = 0

    columns_time = ["time_total", "time_read", "time_merge", "time_write", "time_remove"]
    for column in columns_time:
        df_log[column] = 0.0

    return df_log


