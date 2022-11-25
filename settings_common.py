"""
Set global parameters for the other modules.
You have to modify the parameters in this file depending on your environments, files you want to analysise, and so on.
"""

#coding:utf-8
import os
import numpy as np
import path_info

##
number_fits_for_debug = path_info.number_fits_for_debug 
object_names = None
save_catalog_stars_light_curve_hdf5    = True
save_catalog_stars_light_curve_pickle  = True
save_catalog_stars_header_pickle       = True
save_detected_sources_light_curve      = False
save_detected_sources_movie            = False
make_directory_target_reference        = True

# setting for log
is_previous_log = False
path_previous_log = "/mnt/tomoe/output_analysis/log/20191118_063341/df_log_20191118_063341.csv"

# settings for merge_rank_devided_files.py
remove_files_splitted = True


# settings for pca
n_member_star_min = 10
n_pca_component = 7
filepath_random = "dim_prob.npz"
index_random_percentile = 1
noise_amp_arr = 1
thr_per_correlation = 50
thr_per_variability = 50

# settings for reshape light curve
remove_light_curve_pickle         = True
remove_catalog_star_header_pickle = False
save_catalog_star_header_hdf5     = True
save_light_curve_hdf5             = True
overwrite_light_curve_hdf5        = True
save_catalog_all_stars_hdf5 = True

# settings for reshape light curve with main_concat

aperture_radii_pix = [5, 7, 10, 14, 18] # This is fiducial value, and should be modified (tuned)
params_sep_bkg_default = {
        "mask": None, 
        "maskthresh":0.0, 
        "bw":64, 
        "bh":64, 
        "fw":3, 
        "fh":3, 
        "fthresh":0.0
        }

params_sep_bkg = params_sep_bkg_default
params_sep_bkg["bw"] = 256
params_sep_bkg["bh"] = 256

# memmap = False in order to read all the data into memory at once.
# This enables to major time to read fits file correctly.
params_fits_open = {
        "mode": "readonly",
        "memmap": False
        }


# parameters for fits_analysis.py
# Todo: optimize following params
N_string_catalog_name = 10
length_time_in_string = 26
width_cutout = 20
buffer_dec_deg = 2.0


use_global_rms_as_error = True # Flase is not implemented now. if false, use local Root-Mean Square for background error
params_sep_detection_default = {
        "mask":None, "minarea":5,
        "filter_kernel": np.array([[1,2,1], [2,4,2], [1,2,1]]), 
        "filter_type":'matched',
        "deblend_nthresh":32, "deblend_cont":0.005, "clean":True,
        "clean_param":1.0, "segmentation_map":False
        }
params_sep_detection = params_sep_detection_default
thresh_sep_detection = 1.5
N_detection_max = int(2e7)

if not use_global_rms_as_error:
    params_sep_detection["segmentation_map"] = True

params_sep_photometry_default = {
        "mask":None,
        "maskthresh":0.0,
        "bkgann":None,
        "subpix":5,
        "axes_multiply":6.0, # default value of SExtractor
        "k_kron_radius":2.5, # default value of SExtractor (PHOT_AUTOPARAMS 2.5, 3.5)
        "minimum_diameter":3.5
        }
params_sep_photometry = params_sep_photometry_default
#  params_sep_photometry["subpix"] = 0 # exact overlap is calculated

