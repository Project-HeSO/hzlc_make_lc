"""
Reshape light curve data devided by FITS frame to the data of each star.
Read light curve data of ``light_curve_TMQ....pickle`` and ``catalog_stars_TMQ....pickle``.
Output light curve of each star to HDF5 file (optional).

There are 2 main functions, ``main_concat`` and ``main_append``.

* ``main_concat`` is fast with MPI-parallelization but cannot handle large light curve data (~< 10GB @ 64GB memory size).
  This also returns light curves of stars in the observed data.
* ``main_append`` is slow because MPI-parallelization is disabled but can handle large light curve data.
  light curves of stars in the observed data are saved to HDF5 files.


Author: Kojiro Kawana
"""
from settings_path import *
from settings_common import *

    
import os
import time
import datetime
import glob
import numpy as np
import pandas as pd
import h5py
from mpi4py import MPI

import pickle

import utils
from errors_defined import *
import read_output
import logging
import manage_job
import prepare_run
import catalog_star_history


def output_light_curve_hdf5(df_light_curve, catalog_info, date_obs, directory, overwrite=False):
    """
    Output HDF5 file of light curve of a star.

    If settings.make_directory_target_reference == True, additionaly make symbolic link under target/refernce directory.

        e.g. ``light_curve/target/light_curve_gaia_0000000001.hdf5 -> light_curve/light_curve_gaia_0000000001.hdf5``

    arguments
    =========
    df_light_curve : pandas.DataFrame
        Light curve talbe of the star
    catalog_info: dict-like or pandas.Series
        Catalog information of the star
    date_obs    : str
        observed date: e.g. 20200401
    directory : str
        Directory path where the HDF5 file is output.
    overwrite : bool, optional
        Whether overwrite the HDF5 file if already exists.

    returns
    =======
    output_path : str
        filepath of HDF5 file output
    """

    if type(catalog_info) == pd.core.series.Series:
        pass
    else:
        catalog_info = pd.Series(catalog_info)

    filename = "light_curve_" + str(catalog_info["catalog_name"]) + "_" + str(catalog_info["source_id"]) + ".hdf5"
    output_path = os.path.join(directory, filename)

    if os.path.exists(output_path):
        with h5py.File(output_path, mode="a") as hf:
            if overwrite:
                del(hf["header"], hf[date_obs])
                catalog_info.to_hdf(output_path, key="header", mode="a")
                df_light_curve.to_hdf(output_path, key=date_obs, mode="a", format="table")
            else:
                df_light_curve.to_hdf(output_path, key=date_obs, mode="a", format="table", append=True)
                #  if date_obs in hf.keys():
                #      raise HDF5WriteError("File {:} already exists".format(output_path))
                #  else:
                #      df_light_curve.to_hdf(output_path, key=date_obs, mode="a", format="table", append=True)
    else:
        catalog_info.to_hdf(output_path, key="header", mode="a")
        df_light_curve.to_hdf(output_path, key=date_obs, mode="a", format="table")

    if make_directory_target_reference:
        if catalog_info["is_target"]:
            path_symlink = os.path.join(os.path.dirname(output_path), "target",    os.path.basename(output_path))
        else:
            path_symlink = os.path.join(os.path.dirname(output_path), "reference", os.path.basename(output_path))
        if not os.path.exists(path_symlink):
            os.symlink(output_path, path_symlink)

    return output_path


def mpi_scatter_with_escape(list_scattered, logger, root=0, directory_output="/blacksmith/tmp"):
    """
    Scatter given list from root MPI process to every MPI process.
    First try to scatter with mpi4py scatter. If failed, for example, because the list scattered is too large, 
    escape the error by writing the entry of the list to ``directory_output``, and then each MPI process read the output.

    arguments
    =========
    list_scattered : list
        List scattered. Length must be same as the size of the MPI processes.
    logger : logging.Logger
        Used to print log in cases where MPI scatter is failed.
    root : int
        Rank of root process scattering the list.
    directory_output : str
        Directory path where temporary files are output.

    returns
    =======
    entry_of_my_rank : object (type of each entry of list_scattered)
        == list_scattered[my_rank]
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size == 0:
        entry_of_my_rank= list_scattered[0]
        return entry_of_my_rank

    entry_of_my_rank = None

    try:
        entry_of_my_rank = comm.scatter(list_scattered, root=root)
        logger.info(str(entry_of_my_rank[:2]))

        if (rank != root) and (entry_of_my_rank == "Error"):
            raise OverflowError

    except OverflowError as e:
        if rank == root:
            logger.warning("MPIScatterWaning! Switch to scatter by writing tmp files.\n{:}".format(e))
            signal_error = ["Error"] * size
            comm.scatter(signal_error, root=root)

        else: pass # enter here by receiving eomm.scatter(signal_error, root=root)

        if rank == root:
            os.makedirs(directory_output, exist_ok=True)
            for i, entry_of_my_rank in enumerate(list_scattered):
                path_tmp_output = os.path.join(directory_output, "tmp_rank"+str(i)+".pickle")
                with open(path_tmp_output, "wb") as f:
                    pickle.dump(entry_of_my_rank, f)
            del(list_scattered) # global variant of list_scattered is not deleted
        comm.barrier()

        path_tmp_output = os.path.join(directory_output, "tmp_rank"+str(rank)+".pickle")
        with open(path_tmp_output, "rb") as f:
            entry_of_my_rank = pickle.load(f)
        os.remove(path_tmp_output)

    return entry_of_my_rank


def mpi_gather_with_escape(entry_gathered, logger, root=0, directory_output="/blacksmith/tmp"):
    """
    Gather given entry from every MPI process to root MPI process.
    First try to gather with mpi4py gather. If failed, for example, because the entry is too large, 
    escape the error by writing the entry to ``directory_output``, and then root MPI process reads the output.

    arguments
    =========
    entry_gathered : object
        Entry to be gathered.
    logger : logging.Logger
        Used to print log in cases where MPI gather is failed.
    root : int
        Rank of root process gathering the list.
    directory_output : str
        Directory path where temporary files are output.

    returns
    =======
    list_of_entires: list or None
        For procresses where rank != root, return None.
        Otherwise, this is the list of ``entry_gathered`` owned by each MPI process. The Length is same as the size of the MPI processes.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size == 0:
        list_of_entires = [entry_gathered]
        return list_of_entires

    try:
        list_of_entires = comm.gather(entry_gathered, root=root)
        logger.info(str(entry_of_my_rank[:2]))

        if (rank != root) and (entry_of_my_rank == "Error"):
            # Todo write below
            pass

    except Exception as e:
        if rank == root:
            logger.warning("MPIGatherWaning! Switch to gather by writing tmp files.\n{:}".format(e))

            signal_error = ["Error"] * size
            comm.gather(signal_error, root=root)

        else: pass # enter here by receiving eomm.scatter(signal_error, root=root)

        path_tmp_output = os.path.join(directory_output, "tmp_rank"+str(rank)+".pickle")
        if rank == root:
            os.makedirs(directory_output, exist_ok=True)
        comm.barrier()

        with open(path_tmp_output, "wb") as f:
            pickle.dump(entry_gathered, f)
        del(entry_gathered)
        comm.barrier()

        if rank == root:
            list_of_entires = []
            for i in range(size):
                path_tmp_output = os.path.join(directory_output, "tmp_rank"+str(i)+".pickle")
                with open(path_tmp_output, "rb") as f:
                    list_of_entires.append(pickle.load(path_tmp_output, f))
                os.remove(path_tmp_output)
        else:
            list_of_entires = None

    return list_of_entires


def main_append():
    """
    Main function (MPI-parallel disabled).
    Every time appending data to HDF5 files of star light curve.

    return
    ======
    df_catalog_all_stars : DataFrame
        Catalog data of stars observed in the observed date.

    Todo
    ====
    * supoort MPI-parallel.
      Now do not support in order to avoid write HDF5 file by multiple processes at the same time.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if remove_catalog_star_header_pickle and (not save_catalog_all_stars_hdf5):
        raise ValueError("Settings is wrong!\nIt is not allowed to set, remove_catalog_star_header_pickle  == True & save_catalog_all_stars_hdf5 == False")
        return
    if size > 1:
        raise ValueError("main_append does not support MPI-parallelization!")
        return

    run_start_time = prepare_run.get_start_time_str()
    logger = prepare_run.prepare_logger_merge(os.path.join(directory_log_reshape_light_curve, date_obs), run_start_time)
    logger.info("Start processing\n")

    files_light_curve_pickle = utils.get_fits_frame_file_list(date_obs, dir_output["light_curve_pickle"])
    files_catalog_pickle     = utils.get_fits_frame_file_list(date_obs, dir_output["catalog_pickle"])

    frame_ids  = [file.split("_TMQ")[1][:-7] for file in files_light_curve_pickle]
    frame_ids2 = [file.split("_TMQ")[1][:-7] for file in files_catalog_pickle]
    if not np.prod(frame_ids == frame_ids2):
        logger.exception("Catalog pickle files do not match with light curve pickle files!")
        raise ValueError("Catalog pickle files do not match with light curve pickle files!")

    frame_ids_relabeled = [utils.replace_quarter_in_frame_id(id) for id in frame_ids]
    frame_ids = np.array(frame_ids)
    frame_ids_sorted_by_time = frame_ids[np.argsort(frame_ids_relabeled)]

    files_light_curve_pickle_sorted_by_time = [os.path.join(dir_output["light_curve_pickle"], "light_curve_TMQ"  +frame_id+".pickle") for frame_id in frame_ids_sorted_by_time]
    files_catalog_pickle_sorted_by_time     = [os.path.join(dir_output["catalog_pickle"]    , "catalog_stars_TMQ"+frame_id+".pickle") for frame_id in frame_ids_sorted_by_time]

    df_log = pd.DataFrame(files_catalog_pickle_sorted_by_time, columns=["filepath_catalog"])
    df_log["t_read_light_curve"] = 0.0
    df_log["t_calc_overlap"]     = 0.0
    df_log["t_output_hdf5"]      = 0.0

    dfs_catalog = []
    for i, fpath in enumerate(files_catalog_pickle_sorted_by_time):
        tstart = time.time()
        df_catalog = read_output.read_catalog_stars_pickle(fpath)
        df_catalog["frame_id"] = os.path.basename(fpath).split("_")[-1][:-7] # TMQ...
        dfs_catalog.append(df_catalog)
        df_log.loc[i, "t_read_catalog"] = time.time() - tstart
    df_catalog_concatenated = pd.concat(dfs_catalog)
    subset = ["catalog_name", "source_id"]
    df_catalog_unique = catalog_star_history.drop_duplicate_in_catalog_stars(df_catalog_concatenated, subset=subset)
    logger.info("time read catalog: {:}".format(time.time() - tstart))


    if save_catalog_star_header_hdf5:
        tstart = time.time()
        fpath = os.path.join(directory_catalog_star_headers_out, "catalog_stars_" + str(date_obs) + ".hdf5")
        os.makedirs(directory_catalog_star_headers_out, exist_ok = True)
        df_catalog_concatenated.to_hdf(fpath, key="/data", mode="w", format="table")
        logger.info("time output catalog HDF5: {:}".format(time.time() - tstart))


    columns_output_header = ["catalog_name", "source_id", "catalog_mag", "is_target", "ra", "dec"]
    df_catalog_output = df_catalog_unique[columns_output_header]
    if overwrite_light_curve_hdf5:
        tstart = time.time()
        df_catalog_unique_output = df_catalog_unique[columns_output_header]
        logger.info("Start removing light curve data in HDF5 files already exist")
        Nsize = df_catalog_unique_output.shape[0]

        for i, catalog_info in df_catalog_output.iterrows():
            if (i%100 == 0):
                logger.info("Star number i={:9d} / {:9d}".format(i, Nsize))
            fpath = os.path.join(directory_light_curve_out, "light_curve_" + str(catalog_info["catalog_name"]) + "_" + str(catalog_info["source_id"]) + ".hdf5")
            if os.path.exists(fpath):
                with h5py.File(fpath, mode="a") as hf:
                    if date_obs in hf.keys():
                        del(hf[date_obs])
                    catalog_info.to_hdf(fpath, key="header", mode="a")
        logger.info("time removing light curve data in HDF5 files already exist: {:}".format(time.time() - tstart))


    if save_light_curve_hdf5:
        os.makedirs(directory_light_curve_out, exist_ok=True)
        if make_directory_target_reference:
            os.makedirs(os.path.join(directory_light_curve_out, "target"   ), exist_ok=True)
            os.makedirs(os.path.join(directory_light_curve_out, "reference"), exist_ok=True)

    # store last index for reset index
    length_of_previous_light_curves = np.zeros(df_catalog_unique.shape[0], dtype=int)

    Nsize = len(files_light_curve_pickle_sorted_by_time)
    filepath_log = os.path.join(directory_log_reshape_light_curve, date_obs, run_start_time, "df_log_per_fits.csv")
    pd.DataFrame(df_log.columns).T.to_csv(filepath_log, mode="a", sep=" ", header=None)


    for i, fpath in enumerate(files_light_curve_pickle_sorted_by_time):
        if (i%100 == 0):
            logger.info("FITS frame number i={:5d} / {:5d}".format(i, Nsize))

        if save_light_curve_hdf5:
            tstart = time.time()
            df_light_curve = read_output.read_light_curve_pickle(fpath)
            df_log.loc[i, "t_read_light_curve"] = time.time() - tstart

            tstart = time.time()
            df_catalog_sorted = dfs_catalog[i].sort_values(subset).reset_index()
            star_ids_in_date = utils.extract_overlap_between_2_dataframes(df_catalog_unique, df_catalog_sorted, subset=subset, keep="last").to_numpy()
            df_log.loc[i, "t_calc_overlap"] = time.time() - tstart

            tstart = time.time()
            length_time_axis = df_light_curve.loc[0].shape[0]

            for j, star_id_in_date in enumerate(star_ids_in_date):
                star_id_in_frame = df_catalog_sorted.loc[j, "star_index"]
                df_lc = df_light_curve.loc[star_id_in_frame]
                time_index_start = length_of_previous_light_curves[star_id_in_date]
                df_lc.set_index(np.arange(time_index_start, time_index_start + length_time_axis), drop=True, inplace=True)
                output_light_curve_hdf5(df_lc, df_catalog_output.loc[star_id_in_date],
                                        date_obs, directory_light_curve_out, overwrite=False)

            length_of_previous_light_curves[star_ids_in_date] += length_time_axis
            df_log.loc[i, "t_output_hdf5"] = time.time() - tstart

            pd.DataFrame(df_log.loc[i]).T.to_csv(filepath_log, mode="a", sep=" ", header=None)

    if remove_catalog_star_header_pickle:
        __ = [os.remove(fpath) for fpath in files_catalog_pickle]
    if remove_light_curve_pickle:
        __ = [os.remove(fpath) for fpath in files_light_curve_pickle]

    #  df_log.to_pickle(filepath_log.replace(".csv", ".pickle"))
    logger.info("Finish processing\n")

    return df_catalog_concatenated



def main_concat():
    """
    Main function (MPI-parallel enabled).
    This is faster than main_append, but this requires large memory in order to store all the light curve data in an observed date.

    For example, 64GB memory is not enough for the 20191004 data (Total file size of light_curve_TMQ....pickle is 20 GB)
    (If unpickle the light curve data, 28GB is used, and then a fator of ~3? larger memory is required)

    return
    ======
    df_catalog_all_stars : DataFrame
        Catalog data of stars observed in the observed date.
    list_light_curve: list of DataFrame
        Length = number of stars observed in the observed date.
        Each entry is light curve of a star.
    list_frame_history_of_star : list of list of str
        Length = number of stars observed in the observed date.
        Each entry is history of a star, i.e. in which frames the star has been.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    run_start_time = prepare_run.get_start_time_str()
    logger = prepare_run.prepare_logger_merge(os.path.join(directory_log_reshape_light_curve, date_obs), run_start_time)
    logger.info("Start processing\n")

    if (rank == 0):
        files_light_curve_pickle = utils.get_fits_frame_file_list(date_obs, dir_output["light_curve_pickle"])
        files_catalog_pickle     = utils.get_fits_frame_file_list(date_obs, dir_output["catalog_pickle"])

        frame_ids  = [file.split("_TMQ")[1][:-7] for file in files_light_curve_pickle]
        frame_ids2 = [file.split("_TMQ")[1][:-7] for file in files_catalog_pickle]
        if not np.prod(frame_ids == frame_ids2):
            logger.exception("Catalog pickle files do not match with light curve pickle files!")
            raise ValueError("Catalog pickle files do not match with light curve pickle files!")

        frame_ids_relabeled = [utils.replace_quarter_in_frame_id(id) for id in frame_ids]
        frame_ids = np.array(frame_ids)
        frame_ids_sorted_by_time = frame_ids[np.argsort(frame_ids_relabeled)]

        files_light_curve_pickle_sorted_by_time = [os.path.join(dir_output["light_curve_pickle"], "light_curve_TMQ"  +frame_id+".pickle") for frame_id in frame_ids_sorted_by_time]
        files_catalog_pickle_sorted_by_time     = [os.path.join(dir_output["catalog_pickle"]    , "catalog_stars_TMQ"+frame_id+".pickle") for frame_id in frame_ids_sorted_by_time]

        df_log = pd.DataFrame(files_catalog_pickle_sorted_by_time, columns=["filepath_catalog"])
        df_log["t_read_light_curve"] = 0.0
        df_log["t_calc_overlap"]     = 0.0
        df_log["t_append_list"]      = 0.0

        dfs_catalog = []
        for i, fpath in enumerate(files_catalog_pickle_sorted_by_time):
            tstart = time.time()
            df_catalog = read_output.read_catalog_stars_pickle(fpath)
            df_catalog["frame_id"] = os.path.basename(fpath).split("_")[-1][:-7] # TMQ...
            dfs_catalog.append(df_catalog)
            df_log.loc[i, "t_read_catalog"] = time.time() - tstart
        df_catalog_concatenated = pd.concat(dfs_catalog)
        df_catalog_concatenated.reset_index(inplace=True)
        df_catalog_all_stars = df_catalog_concatenated[["catalog_name", "source_id"]].drop_duplicates()
        df_catalog_all_stars = df_catalog_concatenated.loc[df_catalog_all_stars.index,
                                            ["catalog_name", "source_id", "ra", "dec", "catalog_mag", "is_target"]]
        df_catalog_all_stars.sort_values(["catalog_name", "source_id"], inplace=True, ignore_index=True)
        logger.info("time read catalog: {:}".format(time.time() - tstart))

        tstart = time.time()
        fpath = os.path.join(directory_catalog_star_headers_out, "catalog_stars_" + str(date_obs) + ".hdf5")
        os.makedirs(directory_catalog_star_headers_out, exist_ok = True)
        df_catalog_concatenated.to_hdf(fpath, key="/data", mode="w", format="table")
        logger.info("time output catalog HDF5: {:}".format(time.time() - tstart))

        list_light_curve_raw       = [[] for i in range(df_catalog_all_stars.shape[0])]
        list_frame_history_of_star = [[] for i in range(df_catalog_all_stars.shape[0])]


        Nsize = len(files_light_curve_pickle_sorted_by_time)
        for i, fpath in enumerate(files_light_curve_pickle_sorted_by_time):
            if (i%100 == 0):
                logger.info("FITS frame number i={:5d} / {:5d}".format(i, Nsize))
            tstart = time.time()
            df_light_curve = read_output.read_light_curve_pickle(fpath)
            df_log.loc[i, "t_read_light_curve"] = time.time() - tstart

            tstart = time.time()
            subset = ["catalog_name", "source_id"]
            df_catalog_sorted = dfs_catalog[i].sort_values(subset).reset_index()
            star_ids = utils.extract_overlap_between_2_dataframes(df_catalog_all_stars, df_catalog_sorted,
                                                            subset=subset, keep="last")
            df_log.loc[i, "t_calc_overlap"] = time.time() - tstart

            tstart = time.time()
            for j, star_id in enumerate(star_ids):
                list_light_curve_raw[star_id].append(df_light_curve.loc[df_catalog_sorted.loc[j, "star_index"]])
                list_frame_history_of_star[star_id].append(frame_ids_sorted_by_time[i])
            df_log.loc[i, "t_append_list"] = time.time() - tstart

        filepath_log = os.path.join(directory_log_reshape_light_curve, date_obs, run_start_time, "df_log_per_fits.pickle")
        df_log.to_pickle(filepath_log)


        # prepare columns for log in df_catalog_all_stars
        df_catalog_all_stars["rank_assigned"] = 0
        df_catalog_all_stars["t_concat"]      = 0.0
        df_catalog_all_stars["t_output_hdf5"] = 0.0
        df_catalog_stars_this_rank = [[] for i in range(size)]

        Nframes_of_star = [len(tmp) for tmp in list_frame_history_of_star]
        rank_assigned = manage_job.assign_jobs_to_each_mpi_process_merge(Nframes_of_star, size)
        list_light_curve = [[] for i in range(size)]
        for i in range(size):
            mask = rank_assigned == i
            df_catalog_stars_this_rank[i] = df_catalog_all_stars[mask]
            list_light_curve[i] = [light_curve_raw for flag, light_curve_raw in zip(mask, list_light_curve_raw) if flag]
        del(list_light_curve_raw)

        logger.info("time assign MPI job: {:}".format(time.time() - tstart))

    elif rank != 0:
        list_light_curve           = None
        df_catalog_stars_this_rank = None

    tstart = time.time()
    list_light_curve = mpi_scatter_with_escape(list_light_curve, logger, root=0, directory_output=directory_tmp_output)
    if rank == 0:
        logger.info("time MPI scatter light curve: {:}".format(time.time() - tstart))

    tstart = time.time()
    df_catalog_stars_this_rank = comm.scatter(df_catalog_stars_this_rank, root=0)
    #  df_catalog_stars_this_rank = mpi_scatter_with_escape(df_catalog_stars_this_rank, logger, root=0, directory_output=directory_tmp_output)
    if rank == 0:
        logger.info("time MPI scatter catalog: {:}".format(time.time() - tstart))

    # concatenate DataFrame
    logger.info("Start concat (+ output light curve HDF5)")
    df_catalog_stars_this_rank.reset_index(drop=True, inplace=True)
    Nstar_this_rank = len(list_light_curve)
    logger.info("# stars in this rank N={:d}".format(Nstar_this_rank))
    for i, dfs_light_curve in enumerate(list_light_curve):
        if (i%100 == 0):
            logger.info("Star number i={:9d} / {:9d}".format(i, Nstar_this_rank))
        tstart = time.time()
        list_light_curve[i] = pd.concat(dfs_light_curve).reset_index(drop=True)
        df_catalog_stars_this_rank.loc[i, "t_concat"] = time.time() - tstart

        if save_light_curve_hdf5:
            tstart = time.time()
            os.makedirs(directory_light_curve_out, exist_ok=True)
            if make_directory_target_reference:
                os.makedirs(os.path.join(directory_light_curve_out, "target"   ), exist_ok=True)
                os.makedirs(os.path.join(directory_light_curve_out, "reference"), exist_ok=True)
            try:
                output_light_curve_hdf5(list_light_curve[i], df_catalog_stars_this_rank.loc[i], date_obs, directory_light_curve_out, overwrite=True)
                df_catalog_stars_this_rank.loc[i, "t_output_hdf5"] = time.time() - tstart
            except Exception as e:
                filename = "light_curve_" + str(df_catalog_stars_this_rank.loc[i, "catalog_name"]) + "_" + str(df_catalog_stars_this_rank.loc[i, "source_id"]) + ".hdf5"
                output_path = os.path.join(directory_light_curve_out, filename)
                logger.exception("HDF5WriteError while processing {:}\nSkip output this file\n{:}".format(output_path , e))
                df_catalog_stars_this_rank.loc[i, "t_output_hdf5"] = time.time() - tstart
                continue
    logger.info("Finish concat (+ output light curve HDF5)")

    # MPI gather
    tstart = time.time()
    list_light_curve = mpi_gather_with_escape(list_light_curve, logger, root=0, directory_output=directory_tmp_output)
    if rank == 0:
       list_light_curve = utils.concatenate_list(list_light_curve)
       logger.info("time MPI gather light curve: {:}".format(time.time() - tstart))

    tstart = time.time()
    list_df_catalog_stars = mpi_gather_with_escape(df_catalog_stars_this_rank, logger, root=0, directory_output=directory_tmp_output)
    if rank == 0:
        df_catalog_all_stars = pd.concat(list_df_catalog_stars).reset_index(drop=True)
        logger.info("time MPI gather catalog: {:}".format(time.time() - tstart))


    if rank == 0:
        if remove_light_curve_pickle:
            __ = [os.remove(fpath) for fpath in files_catalog_pickle_sorted_by_time]
        filepath_log = os.path.join(directory_log_reshape_light_curve, date_obs, run_start_time, "df_log_stars.pickle")
        df_catalog_all_stars.to_pickle(filepath_log)
        df_catalog_all_stars = df_catalog_all_stars[["catalog_name", "source_id", "ra", "dec", "catalog_mag", "is_target"]]
        logger.info("Finish processing\n")

        #  debug
        df_catalog_all_stars.to_pickle("df_catalog_all_stars.pickle")
        with open("list_light_curve.pickle", "wb") as f:
            pickle.dump(list_light_curve, f)
        with open("list_frame_history_of_star.pickle", "wb") as f:
            pickle.dump(list_frame_history_of_star, f)

        return df_catalog_all_stars, list_light_curve, list_frame_history_of_star
    else:
        logger.info("Finish processing\n")
        return


if __name__ == "__main__":
    main_append()

