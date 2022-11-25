"""
Modules for the preparation of output directories and initial settings

Author: Kojiro Kawana

Todo
====
* Add Slack alert to looger?
"""
# coding: utf-8
from mpi4py import MPI
import os
import sys
import datetime
import logging
import shutil
from settings_common import *
from settings_path import *

def get_start_time_str():
    """
    Return start time of this pipeline in string format. (e.g. 20200401_120000)
    The time is synchronized for all the MPI processes.

    returns
    =======
    time_run_start : str
    """
    comm = MPI.COMM_WORLD

    if (comm.Get_rank() == 0):
        time_run_start = "{:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    else:
        time_run_start = None
    time_run_start = comm.bcast(time_run_start, root=0)
    return time_run_start


def prepare_logger(dirs_output, time_start):
    """
    Prepare logger where level and hanlers are set.
    Also prepare log directory and log files.

    Log files are as follows:

        * $(dirs_output["log"])/$(time_start) : e.g. ./log/20200401_120000/...

            * df_log_$(time_start).csv
            * stdouterr_$(time_start).txt
            * settings_$(time_start).py: copy of ``settings.py`` used for this run

    arguments
    =========
    dirs_output : dict
        directoris where log, light_curve, movies are output
    time_start : str
        start time of the run

    returns
    =======
    logger : logging.Logger
    filepath_log : str
        filepath of df_log_$(time_stat).csv
    """
    comm = MPI.COMM_WORLD

    log_dir = os.path.join(dirs_output["log"], date_obs)

    filepath_log           = os.path.join(log_dir, time_start, "df_log_{:}.csv".format(time_start))
    filepath_log_stdouterr = os.path.join(log_dir, time_start, "stdouterr_{:}.txt".format(time_start))
    filepath_log_settings_common  = os.path.join(log_dir, time_start, "settings_common_{:}.py".format(time_start))
    filepath_log_settings_path  = os.path.join(log_dir, time_start, "settings_path_{:}.py".format(time_start))

    if (comm.Get_rank() == 0):
        for key, dir_ in dirs_output.items():
            if key == "log":
                os.makedirs(os.path.join(log_dir, time_start), exist_ok=True)
            elif key == "detected_sources":
                for tmp in ["light_curve", "movie"]:
                    dir2_ = os.path.join(dir_, date_obs, tmp)
                    os.makedirs(dir2_, exist_ok=True)
            else:
                os.makedirs(dir_, exist_ok=True)
    comm.Barrier()

    log_format = logging.Formatter('%(asctime)s %(levelname)s %(filename)s (%(lineno)s) rank {:3d} %(message)s'.format(comm.Get_rank()))

    log_file_handler = logging.FileHandler(filepath_log_stdouterr)
    log_file_handler.setLevel(logging.DEBUG)
    log_file_handler.setFormatter(log_format)

    log_stdouterr_handler = logging.StreamHandler()
    log_stdouterr_handler.setLevel(logging.DEBUG)
    log_stdouterr_handler.setFormatter(log_format)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(log_file_handler)
    logger.addHandler(log_stdouterr_handler)

    if (comm.Get_rank() == 0):
        shutil.copy("./settings_common.py", filepath_log_settings_common)
        shutil.copy("./settings_path.py", filepath_log_settings_path)
        # copy the exec command to the stdouterr log file
        with open(filepath_log_stdouterr, mode="a") as f:
            f.write("mpiexec -np {:d} python".format(comm.Get_size()))
            for argv in (sys.argv):
                f.write(" {:}".format(argv))
            f.write("\n")
    comm.Barrier()

    return logger, filepath_log


###########################################
### For merge_rank_devided_files.py #######
##########################################

def prepare_logger_merge(log_dir, time_start):
    """
    Prepare logger where level and hanlers are set.
    Also prepare log directory and log files.

    Log files are as follows:

        * $(dirs_output["log"])/$(time_start) : e.g. ./log/20200401_120000/...

            * stdouterr_$(time_start).txt
            * settings_$(time_start).py: copy of ``settings.py`` used for this run
            * df_log_lc_$(time_start)_rank{:03d}.pickle
            * df_log_movie_$(time_start)_rank{:03d}.pickle


    arguments
    =========
    log_dir : str
        directory path where log files are saved.
    time_start : str
        start time of the run

    Returns
    =======
    logger : logging.Logger
    """
    comm = MPI.COMM_WORLD

    filepath_log_stdouterr = os.path.join(log_dir, time_start, "stdouterr_{:}.txt".format(time_start))
    filepath_log_settings_common  = os.path.join(log_dir, time_start, "settings_common_{:}.py".format(time_start))
    filepath_log_settings_path  = os.path.join(log_dir, time_start, "settings_path_{:}.py".format(time_start))


    if (comm.Get_rank() == 0):
        os.makedirs(os.path.join(log_dir, time_start), exist_ok=True)
    comm.Barrier()

    log_format = logging.Formatter('%(asctime)s %(levelname)s %(filename)s (%(lineno)s) rank {:3d} %(message)s'.format(comm.Get_rank()))

    log_file_handler = logging.FileHandler(filepath_log_stdouterr)
    log_file_handler.setLevel(logging.DEBUG)
    log_file_handler.setFormatter(log_format)

    log_stdouterr_handler = logging.StreamHandler()
    log_stdouterr_handler.setLevel(logging.DEBUG)
    log_stdouterr_handler.setFormatter(log_format)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(log_file_handler)
    logger.addHandler(log_stdouterr_handler)

    if (comm.Get_rank() == 0):
        shutil.copy("./settings_common.py", filepath_log_settings_common)
        shutil.copy("./settings_path.py", filepath_log_settings_path)
        
        # copy the exec command to the stdouterr log file
        with open(filepath_log_stdouterr, mode="a") as f:
            f.write("mpiexec -np {:d} python".format(comm.Get_size()))
            for argv in (sys.argv):
                f.write(" {:}".format(argv))
            f.write("\n")
    comm.Barrier()

    return logger


def prepare_logger_remove_output(log_dir, time_start):
    """
    Prepare logger where level and hanlers are set.
    Also prepare log directory and log files.

    Log files are as follows:

        * $(dirs_output["log"])/$(time_start) : e.g. ./log/20200401_120000/...

            * stdouterr_remove_files_$(time_start).txt

    arguments
    =========
    log_dir : str
        directory path where log files are saved.
    time_start : str
        start time of the run

    Returns
    =======
    logger : logging.Logger
    """

    filepath_log_stdouterr = os.path.join(log_dir, time_start, "stdouterr_remove_output_{:}.txt".format(time_start))
    os.makedirs(os.path.join(log_dir, time_start), exist_ok=True)

    log_format = logging.Formatter('%(asctime)s %(levelname)s %(filename)s (%(lineno)s) %(message)s')

    log_file_handler = logging.FileHandler(filepath_log_stdouterr)
    log_file_handler.setLevel(logging.DEBUG)
    log_file_handler.setFormatter(log_format)

    log_stdouterr_handler = logging.StreamHandler()
    log_stdouterr_handler.setLevel(logging.DEBUG)
    log_stdouterr_handler.setFormatter(log_format)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(log_file_handler)
    logger.addHandler(log_stdouterr_handler)

    return logger


