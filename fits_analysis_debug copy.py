"""
Process FITS files with WCS and output HDF5 files of movie, light curves, detected sources, and header of FITS files.

The output HDF5 files are splitted by MPI processes except for the HDF5 files of detected sources.
Thus, you have to run ``merge_rank_devided_files.py`` after running this ``fits_analysis.py``.

This process does not handle differential photometry nor light curve analysis (such as Lomb-Scargle periodgram).

Author: Kojiro Kawana

Todo
====
* read catalog from MySQL
* catalog_sources_container => DataFrame
* bad pixel mapを共有してマスクして処理
* cross matchを2回行う。 => deteted_sourcesにも保存
* implement forced photometry with sep
* catalog_sources_container (target/reference stars) については追加でforced photometryを実行
* (優先順位低い) => calc_error_for_detected_objects_local_errorを実装 (segmapを使用)
* (優先順位低い) => calc_time_shift_for_detected_objectsを２通りで実装 (1つはcentroidを使用。この方法は後にcatalog starsにもcatalog-implied (x,y)に摘要。
* (優先順位低い) 　 calc_time_shift_for_detected_objectsをsegmapありの場合は各pixelごとを足し上げる方法で実装。これはcalc_error_for_detected_objects_local_errorと実質的に同じ計算)
* status_analysisの数字割当を修正
"""

# coding: utf-8
from settings_path import *
from settings_common import *

from mpi4py import MPI
import os
import time
import numpy as np
import pandas as pd

from errors_defined import *
import calc_time_shift
import prepare_run
import read_catalog
import manage_job
import read_fits
import light_curve_container
import sep_photometry
import cutout_fits
import utils_for_detected_sources
import utils_to_save_data

def main():
    """
    Main function which analyses FITS files and output light_curve, movie, and fits_header HDF5 files.
    """
    comm = MPI.COMM_WORLD

    str_time_start = prepare_run.get_start_time_str()
    logger, filepath_log = prepare_run.prepare_logger(dir_output, str_time_start)
    logger.info("start processing")

    df_catalog = read_catalog.read_all_sky_catalog_merged(directory_catalog, catalog_name_now)
    list_df_log = manage_job.make_df_log_from_fits_directory_debug(directory_fits_files, number_fits_for_debug, object_names= object_names)

    if is_previous_log:
        if (comm.Get_rank() == 0):
            logger.info("Excludes only FITS files previously processd with status_analysis == 2")
        for df_log in list_df_log:
            df_log = manage_job.check_previous_job_and_exclude_finshed_log(path_previous_log, df_log, filepath_log)
            
    #  i_df_log_st, i_df_log_ed = manage_job.assign_jobs_to_each_mpi_process_single_dir(df_log)
    df_log_merged, job_indexes_of_my_process = manage_job.assign_jobs_to_each_mpi_process(list_df_log)
    df_log = df_log_merged.loc[job_indexes_of_my_process]

    # output header (column names to log file)
    if (comm.Get_rank() == 0):
        pd.DataFrame(df_log.columns).T.to_csv(filepath_log, mode="a", sep=" ", header=None)

    # Delete variables unused hereafter to free memory
    del(str_time_start, list_df_log, df_log_merged, job_indexes_of_my_process)

    #  DEBUG
    #  for i_df_log in df_log.index.to_numpy()[:1]:
    for i_df_log in df_log.index:

        df_log.loc[i_df_log, "status_analysis"] = 1 # in analysing

        tstart_tot = time.time()

        try:
            # open FITS file and read header
            tstart = time.time()
            try:
                hdul, fits_header = read_fits.read_fits_header(df_log.loc[i_df_log, "filepath"], calc_WCS = True, **params_fits_open)
            except FitsReadError as e:
                df_log.loc[i_df_log, "t_open_fits"] = time.time() - tstart
                logger.exception("FitsReadError! filepath: {:}\nSkip this FITS file\n{:}\n".format(df_log.loc[i_df_log, "filepath"], e))
                #  error_i_df_log.append(i_df_log)
                df_log.loc[i_df_log, "t_total"] = time.time() - tstart_tot
                df_log.loc[i_df_log, "status_analysis"] = 10
                pd.DataFrame(df_log.loc[i_df_log]).T.to_csv(filepath_log, mode="a", sep=" ", na_rep="NaN", header=None)
                continue

            df_log.loc[i_df_log, "is_wcs_processed"] = "wcs" in fits_header["PIPELINE"]
            df_log.loc[i_df_log, "t_open_fits"] = time.time() - tstart
            if df_log.loc[i_df_log, "is_wcs_processed"]:
                # read catalog files covering the FITS field
                tstart = time.time()
                #  df_catalog = read_catalog.read_ra_dec_splitted_catalogs(fits_header, directory_catalog)
                df_log.loc[i_df_log, "t_read_catalog"] = time.time() - tstart

                 # pickup cataloged stars in the FITS field
                tstart = time.time()
                df_catalog_picked  = read_catalog.pickup_catalog_stars_in_frame(df_catalog, fits_header, buffer_dec_deg = buffer_dec_deg)
                df_log.loc[i_df_log, "N_catalog_stars"] = df_catalog_picked.shape[0]
                df_log.loc[i_df_log, "t_pick_up_catalog_stars"] = time.time() - tstart 

                # prepare light curve and movie container
                tstart = time.time()
                df_catalog_picked["time_correction_catalog_position[s]"] = calc_time_shift.time_shift_sec([fits_header["NAXIS1"], fits_header["NAXIS2"]], df_catalog_picked["x_catalog_position"], df_catalog_picked["y_catalog_position"]).to_numpy()
                df_catalog_picked.index.names = ["star_index"]

                movie_catalog_stars = {}
                movie_catalog_stars["catalog_position"]  = np.zeros((fits_header["NAXIS3"], df_catalog_picked.shape[0], width_cutout, width_cutout), dtype=np.float32)
                movie_catalog_stars["detected_position"] = np.zeros((fits_header["NAXIS3"], df_catalog_picked.shape[0], width_cutout, width_cutout), dtype=np.float32)

                df_total_light_curve_catalog_stars = [[]] * fits_header["NAXIS3"]

                df_log.loc[i_df_log, "t_prepare_light_curve_container"] = time.time() - tstart

            #  DEBUG
            #  for i_time_index in np.array(range(fits_header["NAXIS3"]))[:2]:
            for i_time_index in range(fits_header["NAXIS3"]):
                try:
                    tstart = time.time()
                    df_light_curve_catalog_stars = light_curve_container.make_light_curve_container_of_1frame_as_DataFrame(df_catalog_picked, fits_header, i_time_index)
                    df_log.loc[i_df_log, "t_prepare_light_curve_container"] += time.time() - tstart

                    # read one snapshot
                    tstart = time.time()
                    raw_fits_snapshot = hdul[0].data[i_time_index]
                    df_log.loc[i_df_log, "t_read_snapshot"] += time.time() - tstart

                    # handle background with sep
                    tstart = time.time()
                    bkg_sep, fits_background_subtracted = sep_photometry.background_subtraction_sep(raw_fits_snapshot, **params_sep_bkg)
                    if df_log.loc[i_df_log, "is_wcs_processed"]:
                        df_light_curve_catalog_stars["bkg_rms_global_1_pix"] = bkg_sep.globalrms
                    if use_global_rms_as_error:
                        bkg_error_sep = bkg_sep.globalrms
                    else:
                        bkg_error_sep = bkg_sep.rms()
                    del(bkg_sep) # to free memory
                    df_log.loc[i_df_log, "t_bkg_subtraction"] += time.time() - tstart

                    if df_log.loc[i_df_log, "is_wcs_processed"]:
                        # cutout FITS for catalog sources
                        tstart = time.time()
                        movie_catalog_stars["catalog_position"][i_time_index] = cutout_fits.cutout_sources_from_FITS(raw_fits_snapshot, df_light_curve_catalog_stars[["x_catalog_position", "y_catalog_position"]].to_numpy(), width_cutout)
                        df_log.loc[i_df_log, "t_cutout_FITS_catalog_sources"] += time.time() - tstart

                    # aperture photometry for catalog sources
                    for r in aperture_radii_pix:
                        tstart = time.time()
                        sep_photometry.aperture_photometry_circle_catalog_stars(fits_background_subtracted, bkg_error_sep, df_light_curve_catalog_stars, r=r, gain=fits_header["GAIN"], inplace=True, **params_sep_photometry)
                        df_log.loc[i_df_log, "t_catalog_aperture_photometry_r"+str(r)] += time.time() - tstart

                    # source detection
                    tstart = time.time()
                    try:
                        if params_sep_detection["segmentation_map"] == True:
                            df_detected_sources, segmap = sep_photometry.source_detection_and_photometry_sep(fits_background_subtracted, bkg_error_sep, fits_header, thresh_sep_detection, N_detection_max=N_detection_max, **params_sep_detection)
                        else:
                            df_detected_sources         = sep_photometry.source_detection_and_photometry_sep(fits_background_subtracted, bkg_error_sep, fits_header, thresh_sep_detection, N_detection_max=N_detection_max, **params_sep_detection)
                    except SourceDetectionError as e:
                        logger.exception("SourceDetectionError while processing {:} i_time_index={:}\nSkip this time_index\n{:}".format(df_log.loc[i_df_log, "filepath"], i_time_index, e))
                        df_log.loc[i_df_log, "status_analysis"] = 12
                        continue
                    df_log.loc[i_df_log, "N_detected_sources_in_all_frames"] += df_detected_sources.shape[0]
                    df_log.loc[i_df_log, "t_detection"] += time.time() - tstart


                    if df_log.loc[i_df_log, "is_wcs_processed"]:
                        # calc ra, dec for detected objects
                        tstart = time.time()
                        df_detected_sources.calc_ra_dec_for_detected_objects(fits_header, inplace=True)
                        df_log.loc[i_df_log, "t_calc_ra_dec_detected"] += time.time() - tstart

                        # cross match
                        tstart = time.time()
                        results = utils_for_detected_sources.cross_match(df_catalog_picked, df_detected_sources)
                        for k, v in results.items():
                            df_light_curve_catalog_stars[k] = v
                        df_detected_sources_cross_matched = df_detected_sources.loc[df_light_curve_catalog_stars["id_detected_source"]]
                        df_log.loc[i_df_log, "t_cross_match"] += time.time() - tstart

                    tstart = time.time()
                    df_detected_sources_lc    = df_detected_sources
                    df_detected_sources_movie = df_detected_sources
                    if df_log.loc[i_df_log, "is_wcs_processed"]:
                        if not save_detected_sources_light_curve:
                            # If one detected sources are cross-matched to multiple catalog stars, df_detected_sources_cross_matched has duplicated rows. => remove them
                            df_detected_sources_lc = utils_for_detected_sources.DetectedSources(df_detected_sources_cross_matched.drop_duplicates())
                        if not save_detected_sources_movie:
                            df_detected_sources_movie = utils_for_detected_sources.DetectedSources(df_detected_sources_cross_matched.drop_duplicates())
                    df_log.loc[i_df_log, "t_cross_match"] += time.time() - tstart

                    # calc FLUX_AUTO error for detected objects
                    tstart = time.time()
                    if use_global_rms_as_error:
                        df_detected_sources_lc.calc_error_for_detected_objects_global_error(fits_header, bkg_error_sep, inplace=True)
                    else:
                        # Todo: implement utils_for_detected_sources.calc_error_for_detected_objects_local_error
                        # df_detected_sources_lc = utils_for_detected_sources.calc_error_for_detected_objects_local_error(df_detected_sources_lc, fits_header, bkg_error_sep, segmap)
                        df_detected_sources_lc = utils_for_detected_sources.calc_error_for_detected_objects_local_error(df_detected_sources_lc, fits_header, bkg_error_sep)
                    df_log.loc[i_df_log, "t_calc_FLUX_AUTO_error"] += time.time() - tstart

                    # time-shift for detected sources with their centroids
                    tstart = time.time()
                    df_detected_sources_lc["utc_frame"] = fits_header["time_exp_start_arr"][i_time_index]
                    df_detected_sources_lc["time_correction_centroid[s]"] = calc_time_shift.time_shift_sec([fits_header["NAXIS1"], fits_header["NAXIS2"]], df_detected_sources_lc["x_centroid"], df_detected_sources_lc["y_centroid"])
                    df_log.loc[i_df_log, "t_time_shift_centroid"] += time.time() - tstart

                    # photometry for detected sources
                    tstart = time.time()
                    sep_photometry.aperture_photometry_auto(fits_background_subtracted, bkg_error_sep, df_detected_sources_lc, fits_header, xy_centers=None, inplace=True, **params_sep_photometry)
                    df_log.loc[i_df_log, "t_kron_photometry_detected"] += time.time() - tstart

                    # aperture photometry for detected sources
                    for r in aperture_radii_pix:
                        tstart = time.time()
                        df_detected_sources_lc.aperture_photometry_circle(fits_background_subtracted, bkg_error_sep, r=r, inplace=True, gain=fits_header["GAIN"], **params_sep_photometry)
                        df_log.loc[i_df_log, "t_detected_aperture_photometry_r"+str(r)] += time.time() - tstart

                    # cutout iamges of detected sources
                    tstart = time.time()
                    if i_time_index == 0:
                        movie_detected_sources = np.zeros(fits_header["NAXIS3"], dtype=object)
                    movie_detected_sources[i_time_index] = cutout_fits.cutout_sources_from_FITS(raw_fits_snapshot, df_detected_sources_movie[["x_centroid", "y_centroid"]].to_numpy(), width_cutout = width_cutout)
                    df_log.loc[i_df_log, "t_cutout_FITS_detected_sources"] += time.time() - tstart

                    if df_log.loc[i_df_log, "is_wcs_processed"]:
                        # save flux and so on into light curve container
                        tstart = time.time()
                        df_light_curve_catalog_stars = light_curve_container.save_detected_sources_into_catalog_stars_DataFrame(df_light_curve_catalog_stars, df_detected_sources_lc.loc[df_detected_sources_cross_matched.index])
                        df_total_light_curve_catalog_stars[i_time_index] = df_light_curve_catalog_stars

                        df_detected_sources_movie["index_for_movie_slice"] = np.arange(df_detected_sources_movie.shape[0])
                        movie_catalog_stars["detected_position"][i_time_index] = movie_detected_sources[i_time_index][df_detected_sources_movie.loc[df_detected_sources_cross_matched.index, "index_for_movie_slice"]]
                        df_detected_sources_movie.drop(columns="index_for_movie_slice", inplace=True)
                        df_log.loc[i_df_log, "t_save_to_catalog_sources"] += time.time() - tstart

                    # output detected sources into HDF5 file
                    if save_detected_sources_light_curve:
                        tstart = time.time()
                        df_detected_sources_lc.output_to_hdf5(dir_output["detected_sources"], fits_header, i_time_index)
                        df_log.loc[i_df_log, "t_output_detected_sources_light_curve"] += time.time() - tstart

                except Exception as e:
                    logger.exception("Unclassified error while processing {:} i_time_index={:}\nSkip this time_index\n{:}".format(df_log.loc[i_df_log, "filepath"], i_time_index, e))
                    df_log.loc[i_df_log, "status_analysis"] = 50
                    continue



            hdul.close()

            if df_log.loc[i_df_log, "is_wcs_processed"]:
                tstart = time.time()
                df_total_light_curve_catalog_stars = pd.concat(df_total_light_curve_catalog_stars)
                df_total_light_curve_catalog_stars.sort_values(["star_index", "utc_frame"], inplace=True)
                for col in light_curve_container._columns_pca:
                    df_total_light_curve_catalog_stars[col] = 0.0
                df_total_light_curve_catalog_stars = df_total_light_curve_catalog_stars[light_curve_container._sorted_columns_light_curve_DataFrame] 
                df_total_light_curve_catalog_stars["frame_id"] = fits_header["FRAME_ID"]
                df_total_light_curve_catalog_stars["exptime"]  = fits_header["EXPTIME1"]
                df_log.loc[i_df_log, "t_reshape_catalog_sources_pandas"] += time.time() - tstart

            tstart = time.time()
            utils_to_save_data.output_fits_header_to_pickle(fits_header, dir_output["fits_header"], logger, df_log, i_df_log)
            df_log.loc[i_df_log, "t_output_fits_header"] += time.time() - tstart

            if save_detected_sources_movie:
                tstart = time.time()
                path_output = os.path.join(dir_output["detected_sources"], date_obs, "movie", "movie_detected_sources_{:}.npy".format(fits_header["FRAME_ID"]))
                np.save(path_output, movie_detected_sources, allow_pickle=True)
                df_log.loc[i_df_log, "t_output_detected_sources_movie"] += time.time() - tstart

            if df_log.loc[i_df_log, "is_wcs_processed"]:
                # output light curves and movie to hdf5
                if save_catalog_stars_light_curve_hdf5:
                    tstart = time.time()
                    utils_to_save_data.output_light_curve_to_hdf5(df_catalog_picked, df_total_light_curve_catalog_stars, dir_output["light_curve"], fits_header, logger, df_log, i_df_log)
                    df_log.loc[i_df_log, "t_output_light_curve"] += time.time() - tstart

                tstart = time.time()
                # (time, N_stars) => (N_stars, time)
                for k, v in movie_catalog_stars.items():
                    movie_catalog_stars[k] = v.swapaxes(0, 1)
                utils_to_save_data.output_movie_to_hdf5(movie_catalog_stars, df_catalog_picked, df_total_light_curve_catalog_stars, dir_output["movie"], fits_header, logger, df_log, i_df_log)
                df_log.loc[i_df_log, "t_output_movie"] += time.time() - tstart

                if save_catalog_stars_header_pickle:
                    tstart = time.time()
                    path_output = os.path.join(dir_output["catalog_pickle"], "catalog_stars_{:}.pickle".format(fits_header["FRAME_ID"]))
                    df_catalog_picked.to_pickle(path_output)
                    df_log.loc[i_df_log, "t_output_catalog_pickle"] += time.time() - tstart
                

                if save_catalog_stars_light_curve_pickle:
                    tstart = time.time()
                    path_output = os.path.join(dir_output["light_curve_pickle"], "light_curve_{:}.pickle".format(fits_header["FRAME_ID"]))
                    df_total_light_curve_catalog_stars.to_pickle(path_output)
                    df_log.loc[i_df_log, "t_output_light_curve_pickle"] += time.time() - tstart

            for column in ["t_output_light_curve", "t_output_movie", "t_output_light_curve_pickle", "t_output_detected_sources_light_curve", "t_output_detected_sources_movie", "t_output_fits_header"]:
                df_log.loc[i_df_log, "t_output_total"] += df_log.loc[i_df_log, column]

            # output log
            df_log.loc[i_df_log, "t_total"] = time.time() - tstart_tot
            if (df_log.loc[i_df_log, "status_analysis"] == 1):
                # if succesfully finished
                df_log.loc[i_df_log, "status_analysis"] = 2
            pd.DataFrame(df_log.loc[i_df_log]).T.to_csv(filepath_log, mode="a", sep=" ", na_rep="NaN", header=None)

        except Exception as e:
            logger.exception("Unclassified error while processing {:}\n{:}".format(df_log.loc[i_df_log, "filepath"], e))
            #  error_i_df_log.append(i_df_log)
            df_log.loc[i_df_log, "t_total"] = time.time() - tstart_tot
            df_log.loc[i_df_log, "status_analysis"] = 50
            pd.DataFrame(df_log.loc[i_df_log]).T.to_csv(filepath_log, mode="a", sep=" ", na_rep="NaN", header=None)
            continue

    logger.info("Finish processing\n")

if __name__ == "__main__":
    main()

