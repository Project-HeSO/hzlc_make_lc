"""
Set global parameters for the other modules.
You have to modify the parameters in this file depending on your environments, files you want to analysise, and so on.
"""

#coding:utf-8
import os
import numpy as np
rootdir = os.path.abspath(os.path.dirname(__file__))

#観測されたfitsファイルを入れるディレクトリ
alpsdir = "/alps2"
alps_dirs = [os.path.join(alpsdir, tmp) for tmp in ["north", "center", "south"]]

file_test = open("date_setting.py", "r")
lines = file_test.readlines()
date_obs = lines[0]
file_test.close()
directory_fits_files = [os.path.join(tmp, date_obs) for tmp in alps_dirs]
if len(directory_fits_files) == 0:
        directory_fits_files = [os.path.join(tmp, date_obs, "raw") for tmp in alps_dirs]

##
number_fits_for_debug = 20000
object_names = None
#object_names = ["TESS1516", "gaia19alg"]
save_catalog_stars_light_curve_hdf5    = True
save_catalog_stars_light_curve_pickle  = True
save_catalog_stars_header_pickle       = True
save_detected_sources_light_curve      = False
save_detected_sources_movie            = False
make_directory_target_reference        = True


#pipelineに必要なデータ類や完了したデータ等を入れるディレクトzfリ
product_out_dir = "/shosoin/product/out_analysis_%s" % date_obs
smithdir = "/shosoin/intermediate_product/out_analysis_%s" % date_obs

dir_output = {}
dir_output["fits_header"]         = os.path.join(smithdir, "fits_header")
dir_output["light_curve"]         = os.path.join(smithdir, "light_curve")
dir_output["light_curve_pickle"]  = os.path.join(smithdir, "light_curve_pickle")
dir_output["catalog_pickle"]      = os.path.join(smithdir, "catalog_pickle")
dir_output["movie"]               = os.path.join(smithdir, "movie")
dir_output["detected_sources"]    = os.path.join(product_out_dir, "detected_sources")
dir_output["log"]                 = os.path.join(product_out_dir, "log")


#  catalog_dir = 
directory_catalog = "/blacksmith/catalog/used_now/"
catalog_name_now = "ohsawa_renewed_20210429.pickle"
is_previous_log = False
path_previous_log = "/mnt/tomoe/output_analysis/log/20191118_063341/df_log_20191118_063341.csv"


# settings for merge_rank_devided_files.py
directory_fits_header_in  = dir_output["fits_header"]
directory_light_curve_in  = dir_output["light_curve"]
directory_movie_in        = dir_output["movie"]
directory_log_merge       = os.path.join(product_out_dir, "log_merge")
directory_fits_header_out = os.path.join(product_out_dir, "fits_header")
directory_light_curve_out = os.path.join(product_out_dir, "light_curve")
directory_movie_out       = os.path.join(product_out_dir, "movie")
remove_files_splitted = True

# settings for pca
directory_log_pca = os.path.join(product_out_dir, "log_pca")
dir_catalog_star = "%s/catalog_star_header/" % product_out_dir 
dir_light_curve = "%s/light_curve/" % product_out_dir 
directory_light_curve_out_pca = "%s/light_curve_pca/" % product_out_dir 
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
directory_log_reshape_light_curve  = os.path.join(product_out_dir, "log_reshape")
directory_catalog_star_headers_out = os.path.join(product_out_dir, "catalog_star_header")
directory_light_curve_out          = os.path.join(product_out_dir, "light_curve")

# settings for reshape light curve with main_concat
# directory_tmp_output = "/blacksmith/tmp"
# settings for photometry

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

