"""
Set global parameters for the other modules.
You have to modify the parameters in this file depending on your environments, files you want to analysise, and so on.
"""

#coding:utf-8
import os
import numpy as np
import path_info
rootdir = os.path.abspath(os.path.dirname(__file__))

## root directory for raw, catalog, & product
alpsdir = path_info.alpsdir
product_dir = path_info.product_dir
catalog_dir =  path_info.catalog_dir 
catalog_name_now = path_info.catalog_name 
path_previous_log = "/mnt/tomoe/output_analysis/log/20191118_063341/df_log_20191118_063341.csv"
date_obs = str(path_info.date_obs)


#Directory for data
alps_dirs = [os.path.join(alpsdir, tmp) for tmp in ["north", "center", "south"]]
directory_fits_files = [os.path.join(tmp, date_obs) for tmp in alps_dirs]
if len(directory_fits_files) == 0:
        directory_fits_files = [os.path.join(tmp, date_obs, "raw") for tmp in alps_dirs]


#Directories for product
product_out_dir = os.path.join(product_dir, "product/out_analysis_%s" % date_obs)
smithdir = os.path.join(product_dir, "intermediate_product/out_analysis_%s" % date_obs)

dir_output = {}
dir_output["fits_header"]         = os.path.join(smithdir, "fits_header")
dir_output["light_curve"]         = os.path.join(smithdir, "light_curve")
dir_output["light_curve_pickle"]  = os.path.join(smithdir, "light_curve_pickle")
dir_output["catalog_pickle"]      = os.path.join(smithdir, "catalog_pickle")
dir_output["movie"]               = os.path.join(smithdir, "movie")
dir_output["detected_sources"]    = os.path.join(product_out_dir, "detected_sources")
dir_output["log"]                 = os.path.join(product_out_dir, "log")


#  Directory for catalog
directory_catalog = os.path.join(catalog_dir ,"catalog/used_now/")
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


# settings for reshape light curve
directory_log_reshape_light_curve  = os.path.join(product_out_dir, "log_reshape")
directory_catalog_star_headers_out = os.path.join(product_out_dir, "catalog_star_header")
directory_light_curve_out          = os.path.join(product_out_dir, "light_curve")
