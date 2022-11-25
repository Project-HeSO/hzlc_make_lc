
from settings_path import *
from settings_common import *
    
import numpy as np
import pickle
import pandas as pd
import copy
import glob
import os 
import re
import h5py
import copy
import time
import logging
from mpi4py import MPI

import astropy
import astropy.units as u
from astropy.time import Time

import read_output
import catalog_star_history
import manage_job
import prepare_run


def get_complete_source_ids(df_catalog_all_stars):
    ''' Obtain the complete source IDs from df_catalog_all_stars

    Args:
        df_catalog_all_stars : 
    
    Returns:
        source_ids_all : list of complete soruce IDs
    
    '''
    source_ids_all = []
    list_of_sourceid = df_catalog_all_stars['source_id']
    list_of_catalogid = df_catalog_all_stars['catalog_name']
    
    for i in range(len(list_of_sourceid)):
        source_ids_all.append(list_of_catalogid[i] + '_' + str(list_of_sourceid[i]))
    
    return source_ids_all


def group_stars_based_on_obs_history(source_ids_all,list_frame_history_of_star):
    ''' Group stars based on their frame ID histories
     
    Args:
        source_ids_all : 
        list_frame_history_of_star : 
    
    Returns:
        dict_of_obs_history : dictionary of the star IDs. The keys are the frame ID histories
   
    '''
    dict_of_obs_history = {}

    for i, source_id in enumerate(source_ids_all):
        fitsids = list_frame_history_of_star[i]
    
        fits_history = ''
        for dmy in fitsids:
            fits_history += 'TMQ' + dmy
        
        if not fits_history in dict_of_obs_history:
            dict_of_obs_history[fits_history] = []

        dict_of_obs_history[fits_history].append(source_id)
    
    return dict_of_obs_history


def store_group_ids_to_df_catalog_all_stars(df_catalog_all_stars, pca_group_dict_index, nonpca_group_dict_index):
    """
    Store grouping info used for PCA to DataFrame of catalog stars.
    arguments
    =========
    df_catalog_all_stars: DataFrame
    pca_group_dict_index: dict
    nonpca_group_dict_index: dict
    returns
    =======
    df_catalog_all_stars: DataFrame
    Column of "pca_group_id" is added.
    Group-IDs of each star are shown.
    Stars analysised with PCA have group-IDs of 1,2,3...
    Stars not-analysised with PCA have group-IDs of -1,-2,-3,...
    """
    df_catalog_all_stars["pca_group_id"] = 0
    for i, indexes in enumerate(nonpca_group_dict_index.values()):
        df_catalog_all_stars.loc[indexes, "pca_group_id"] = - (i + 1)
    for i, indexes in enumerate(pca_group_dict_index.values()):
        df_catalog_all_stars.loc[indexes, "pca_group_id"] = i + 1
    return df_catalog_all_stars


def replace_index_to_star_name(df_catalog_all_stars, dict_of_obs_history_index):
    """
    arguments
    =========
    df_catalog_all_stars
    dict_of_obs_history_index
    returns
    =======
    dict_of_obs_history_star_name
    """
    dict_of_obs_history_star_name = {}
    for key, indexes in dict_of_obs_history_index.items():
        catalog_names = df_catalog_all_stars.loc[indexes, ["catalog_name"]]["catalog_name"]
        source_ids    = df_catalog_all_stars.loc[indexes, ["source_id"]   ]["source_id"]
        names = [catalog_name + "_" + str(source_id) for catalog_name, source_id in zip(catalog_names, source_ids)]
        dict_of_obs_history_star_name[key] = names
    return dict_of_obs_history_star_name


def divide_group_into_pca_and_nonpca(dict_of_obs_history,n_member_star_min = 10):
    ''' Divide the star groups into two; one is being PCA anlysed the other is not. The latter groups consist of smaller number of stars than n_member_star_min.
     
    Args    
        dict_of_obs_history : 
        n_member_star_min : the minimum threshold value of number of stars in a group for PCA
    
    Returns
        pca_group_dict : dictionary of the star IDs to be PCA-ed. The keys are the PCA group name
        nonpca_group_dict : dictionary of the star IDs not to be PCA-ed
       
    '''

    keys_of_major = []
    keys_of_minor = []

    keys = list(dict_of_obs_history.keys())
    for key in keys:
        if len(dict_of_obs_history[key]) > n_member_star_min: 
            keys_of_major.append(key)
        else :
            keys_of_minor.append(key)

    print(keys_of_major)
    print(keys_of_minor)
    nonpca_group_dict = {}
    pca_group_dict = {}

    for key_of_minor in keys_of_minor:
        print(key_of_minor)
        key_splitted = re.split('[TMQ]',key_of_minor)
        key_splitted = [x for x in key_splitted if x != '']
        #print(key_splitted)
        for key in keys_of_major:
            pca_group_dict[key] = dict_of_obs_history[key]
            flag = True
            for dmy in key_splitted: 
                flag = flag*(dmy in key)
            if flag:
                pca_group_dict[key].extend(dict_of_obs_history[key_of_minor])
                break

        if not flag:
            nonpca_group_dict[key_of_minor] = dict_of_obs_history[key_of_minor]

    if len(keys_of_minor) ==0:
        for key in keys_of_major:
            pca_group_dict[key] = dict_of_obs_history[key]

    return pca_group_dict, nonpca_group_dict


def load_light_curves(source_ids,dir_light_curve,date):
    ''' Get list of light curves of stars in a PCA group
    
    Args:
        source_ids:
        dir_light_curve:
        date:
    
    Returns:
        source_ids: 
        list_light_curve:
    '''
    
    list_light_curve = []
    num_rm = []
    for i, dmy in enumerate(source_ids):
        fpath = os.path.join(dir_light_curve, "light_curve_" + dmy + ".hdf5")
        df = read_output.read_LC_hdf5(fpath, date=str(date))
        try:
            df[0]
            list_light_curve.append(df[1][0])
        except :
            num_rm.append(i)
    source_ids = np.delete(source_ids,num_rm,0)
    
    return source_ids, list_light_curve


def pca_for_a_group_and_a_fluxtype(name_of_flux, list_light_curve, source_ids, ncomp, filepath_random, index_random_percentile, noise_amp_arr, thr_per_correlation, thr_per_variability):
    ''' PCA a type of flux in a PCA group, consisting of dividing the group into main and sub group, padding light curves of the stars in the sub-group, SVD, unpadding light curves of the stars in the sub-group.

    Args:
        name_of_flux:  
        list_light_curve: 
        source_ids: 
        ncomp:
        filepath_random:
        noise_amp_arr
        thr_per_correlation
        thr_per_variability

    Returns:
        source_ids:
        flux_arr_unpad:
        time_arr_unpad:

    Todo:
        Refactor arguments using dictionary.
        Add explanations for arguments
    '''
    flux_arr_main,time_arr_main, source_ids_main,flux_arr_sub,time_arr_sub,source_ids_sub,out_int,obs_duration = divide_a_pca_group(name_of_flux, list_light_curve, source_ids)
    flux_arr_pad,time_arr_pad,source_ids,use_svd_flag,padding_mask,pca_mask = padding_light_curves(flux_arr_main, time_arr_main, source_ids_main,flux_arr_sub, time_arr_sub,source_ids_sub,out_int)
    after_subtracted, V_sub_from_lc, sigma_arr = svd_fit_and_subtract_after_padding(flux_arr_pad, obs_duration, use_svd_flag, ncomp, filepath_random, index_random_percentile, noise_amp_arr, thr_per_correlation, thr_per_variability)
    flux_arr_unpad,time_arr_unpad = unpad_light_curve(after_subtracted,time_arr_pad,padding_mask,pca_mask)
    
    return  source_ids,flux_arr_unpad,time_arr_unpad


def divide_a_pca_group(name_of_flux, list_light_curve, source_ids):
    ''' Divide stars in a PCA group into two; ones are with the longest observed duration and the others have shorter observed times. 

    Args:
        name_of_flux:  
        list_light_curve: 
        source_ids: 

    Returns:
        flux_arr_main : array of fluxes of stars with the longest observed duration
        time_arr_main : array of times of stars with the longest observed duration
        source_ids_main : list of source IDs of stars with the longest observed duration
        flux_arr_sub : array of fluxes of stars with a shorter observed duration
        time_arr_sub : array of times of stars with a shorter observed duratio
        source_ids_sub : list of source IDs of stars with a shorter observed duration
        out_int : indices of stars whose light curves need to be padded
        obs_duration : list of observed duration of stars
    '''
    
    flux_arr = []
    time_arr = []
    obs_duration = []

    for i, dmy in enumerate(source_ids):
        flux_arr.append(list(list_light_curve[i][name_of_flux]))
        time_stamps = (list_light_curve[i]["utc_frame"] - list_light_curve[i]["utc_frame"].min()).astype(np.dtype("timedelta64[us]")) * 1e-6
        time_arr.append(list(list_light_curve[i]["utc_frame"]))
        obs_duration.append(len(time_stamps))

    obs_duration_max = np.max(obs_duration)

    flux_arr_sub = []
    time_arr_sub = []
    source_ids_sub = []

    flux_arr = np.array(flux_arr)
    time_arr = np.array(time_arr)

    out_int = []
    for i, flux in enumerate(flux_arr):
        if len(flux) != obs_duration_max:
            out_int.append(i)
            flux_arr_sub.append(flux) 
            time_arr_sub.append(time_arr[i])
            source_ids_sub.append(source_ids[i])
    flux_arr_main = np.delete(flux_arr,out_int,0)
    time_arr_main  = np.delete(time_arr,out_int,0)
    source_ids_main = np.delete(source_ids,out_int,0)

    return flux_arr_main, time_arr_main, source_ids_main,flux_arr_sub, time_arr_sub, source_ids_sub, out_int, obs_duration


def padding_light_curves(flux_arr, time_arr, source_ids,flux_arr_sub, time_arr_sub,source_ids_sub,out_int):
    ''' Padding light curves of stars with a shorter observation duration 

    Args:
        flux_arr:
        time_arr:
        source_ids:
        flux_arr_sub:
        time_arr_sub:
        source_ids_sub:
        out_int:
    
    Returns:
        flux_arr : padded & rearranged flux array
        time_arr : padded & rearranged time array
        source_ids : rearraned list of source ids 
        use_svd_flag : flag for singular value decomposision
        padding_mask : mask the padded data points
        pca_mask : mask the padded stars 
        
    '''
    a_ = list(flux_arr)
    b_ = list(time_arr)
    c_ = list(source_ids)
    pca_mask = np.ones_like(time_arr,dtype=bool)
    padding_mask = []
    
    longest_time_arr = time_arr[0]
    for i, tmp in enumerate(time_arr_sub):
        ttmp = flux_arr_sub[i]
        tttmp = source_ids_sub[i]
        fm = np.mean(ttmp)
        mask = np.ones_like(tmp, dtype=bool)
            
        for j, x in enumerate(longest_time_arr):
            if x not in tmp:
                tmp = np.insert(tmp, j, x)
                ttmp = np.insert(ttmp, j, fm)
                mask = np.insert(mask,j,False)
            
        a_.append(list(ttmp))
        b_.append(list(tmp))
        c_.append(tttmp)
        pca_mask = np.append(pca_mask,False)
        padding_mask.append(mask)

    flux_arr_pad = np.array(a_)
    time_arr_pad = b_
    source_ids = c_
    if out_int:
        use_svd_flag= pca_mask
    else :
        use_svd_flag = np.ones(len(flux_arr),dtype=bool)

    return flux_arr_pad,time_arr_pad,source_ids,use_svd_flag, padding_mask,pca_mask
        

def svd_fit_and_subtract_after_padding(flux_arr, obs_duration, use_svd_flag = None, ncomp=7, filepath_random = "dim_prob.npz", index_random_percentile = 1, noise_amp_arr = 1, thr_per_correlation = 50, thr_per_variability = 30):
    ''' Signular value decomposition of light curves, subtract PCA components with padding some light curves, if necessary. Basically imported from Aizawa pipeline.
     
    Args:
        flux_arr : 
        obs_duration : 
        use_svd_flag : 
        ncomp : number of PCA components being subtracted
        filepath_random : file include radam variables for setting the threshold for pca coefficients 
        index_random_percentile : 
        noise_amp_arr : 
        thr_per_correlation : 
        thr_per_variability : 
        
    
    Returns:
        after_subtracted:
        V_sub_from_lc:
        sigma_arr:
    
    '''
    ## load threshold for pca coefficients
    if os.path.exists(filepath_random):
        data_now = np.load(filepath_random)
        arr_max_nrandom = data_now["arr_max"]
        dims = data_now["dims"]
        dic_threshold = dim_to_threshold_pca_from_random_sphere(arr_max_nrandom[:,index_random_percentile], dims)

    else:
        arr_max_nrandom, percentile, dims = main_dim_threshold(dim=1000)
        dic_threshold = dim_to_threshold_pca_from_random_sphere(arr_max_nrandom[:,index_random_percentile], dims)


    if use_svd_flag is None:
        use_svd_flag = np.ones(len(flux_arr))>0
    
    
    n1, p1 = np.shape(flux_arr)
    flux_med_arr = x_mean(flux_arr)
    flux_div = flux_meddiv_and_subtract_1(flux_arr)
    
    ## Choose less variable stars
    flux_std = np.std(flux_div, axis=1)
    V = flux_std/(noise_amp_arr)
    V = V/np.median(V)
    flag_V_cut = V < np.percentile(V, thr_per_variability)

    
    ## Choose strong correlated stars
    flag_for_used = np.copy(flag_V_cut)
    corr, corr_all = pearson_correlation_for_stars(flux_div[flag_V_cut])
    corr_flag = corr > np.percentile(corr, thr_per_correlation)
    count_star = 0
    
    ## Summarize flags
    for i in range(len(flag_V_cut)):

        if flag_V_cut[i]:
            if corr_flag[count_star]:
                flag_for_used[i] = True
            else:
                flag_for_used [i] = False
            count_star += 1
    
    use_svd_flag = use_svd_flag * flag_for_used #* corr_flag
    
    ## If the number of stars for SVD is 0 or 1, we cannot do PCA
    if len(use_svd_flag[use_svd_flag])<2:
        return flux_arr, None, None

    ## SVD to determine basis vectors
    flux_arr_for_pca = flux_div[use_svd_flag,:]
    n_pca = np.sum(use_svd_flag)
    p_pca = np.shape(flux_arr)[1]    
    U, s ,V_svd = np.linalg.svd(flux_arr_for_pca, full_matrices = True)

    ## Determine basis vectors for subtraction
    coeff_component_star = np.max(np.abs(U.T), axis=1)
    n_comp_from_threshold = find_number_of_consective_one(coeff_component_star < dic_threshold[len(s)])
    n_sub_comp = np.min([ncomp, n_comp_from_threshold])
    
    ## If there is no component subtraction for PCA
    if n_sub_comp == 0:
        return flux_arr, None, None
    
    
    ## Determine diagonal matrix
    max_value = max(n_pca, p_pca) 
    min_value = min(n_pca, p_pca)
    S = np.zeros((max_value , max_value )) ###
    S[:min_value, :min_value] = np.diag(s) ### diag of singular values
    sigma_arr = s*s/p_pca ### significance of each component

    ## Component subtraction
    V_sub_from_lc = V_svd[:n_sub_comp] ## Component for subtraction 
    coeff_V = np.dot(flux_div,V_sub_from_lc.T) ## Coefficient for subtraction 
    sub_component = np.dot(coeff_V, V_sub_from_lc) ## Lightcurve for subtraction 
    pad_weight = np.diag(np.max(obs_duration)/obs_duration)
    sub_component = np.dot(pad_weight,sub_component)
    after_subtracted = (flux_div-sub_component + 1) * flux_med_arr ## Output after subtraction of systematics
    
    print(np.std(after_subtracted))
    return after_subtracted, V_sub_from_lc, sigma_arr

def x_mean(x):
    ''' Calculate mean of an array
    
    Args:
        x : numpy array of shape (n,num_bamd)
        
    Returns:
        x_mean_arr : numpy array of size n
        
    Todo:
        Remove 'for' to make faster
    '''
    n, num_band = np.shape(x)
    x_mean_arr = np.zeros(np.shape(x))
    for i in range(n):
        x_mean_arr[i] = np.mean(x[i])
    
    return x_mean_arr


def flux_sub_mean_func(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    zscore = (x-xmean)
   
    return zscore


def flux_std(x):
    ''' Calculate standard diviation of an array
    
    Args:
        x : numpy array of shape (n,num_bamd)
        
    Returns:
        x_std_arr : numpy array of size n
        
    '''
    n, num_band = np.shape(x)
    x_std_arr = np.zeros(n)
    for i in range(n):
        x_std_arr[i] = np.std(x[i])
    
    return x_std_arr


def flux_meddiv_and_subtract_1(lcs):
    nstar, nt = np.shape(lcs)
    flux_med_arr = x_mean(lcs)
    flux_div = np.zeros(np.shape(lcs))

    for i in range(nstar):
        flux_div[i] = (lcs[i]/flux_med_arr[i])-1

    return flux_div
    
def find_number_of_consective_one(arr):
    count = 0

    for i in range(len(arr)):
        if arr[i] !=1:
            return count
        count+=1
    return count

def pearson_correlation_for_stars(lcs):

    nstar, nt = np.shape(lcs)

    lcs_sub = lcs.T - np.mean(lcs, axis = 1)
    lcs_sub = lcs_sub.T
    lcs_std = np.array(np.std(lcs_sub, axis = 1))
    lcs_sub_div = lcs_sub.T/lcs_std
    lcs_sub_div = lcs_sub_div.T
    corr = np.abs(np.dot(lcs_sub_div, lcs_sub_div.T)/(1.0 * nt))
    corr_med = np.median(corr, axis = 0)
    return corr_med, corr

## Making random_sphere threshold                                                                                                                                                                           
def dim_to_threshold_pca_from_random_sphere(threshold, dims):
    dic_dim_to_thr = {}
    for i in range(len(threshold)):
        dic_dim_to_thr[dims[i]] = threshold[i]
    return dic_dim_to_thr

def random_vectors_on_sphere(N_sample_vectors, dimensions, r=1):
    ret = np.zeros((N_sample_vectors, dimensions), dtype=float)
    for i in range(N_sample_vectors):
        x = np.random.randn(dimensions)
        r = np.linalg.norm(x)
        if r != 0.:
            ret[i] = x/r
    return ret

def return_max_arr(dimensions, N_sample_vectors=100000, percentile = [50,90,95, 99], file_name="dim_prob"):
    vecs_ = random_vectors_on_sphere(N_sample_vectors, dimensions)
    vec_max = np.max(np.abs(vecs_), axis=1)
    return np.percentile(vec_max, percentile)

def main_dim_threshold(dim=5):
    percentile = np.array([50,90,95, 99])
    arr_max_nrandom = []
    dims = np.arange(2,dim)
    for dim in dims:
        if dim %10==0:
            print(dim)
        arr_percent = return_max_arr(dim, percentile = percentile)
        arr_max_nrandom.append(arr_percent)
    arr_max_nrandom = np.array(arr_max_nrandom)
    np.savez(file="dim_prob", arr_max = arr_max_nrandom, percent = percentile, dims = dims)


    return arr_max_nrandom, percentile, dims
    
def unpad_light_curve(after_subtracted,time_arr_pad,padding_mask,pca_mask):
    ''' Unpad PCA-ed light curves   

    Args:
        after_subtracted:
        time_arr_pad:
        padding_mask:
        pca_mask:

    Returns:
        flux_arr_unpad:
        time_arr_unpad:
    '''
    
    if len(padding_mask) == 0:
        return after_subtracted,time_arr_pad
    
    else :
        flux_arr_padded = after_subtracted[-len(padding_mask):]
        flux_arr_unpad = list(after_subtracted[pca_mask])
        
        time_arr_pad = np.array(time_arr_pad)
        time_arr_padded = time_arr_pad[-len(padding_mask):]
        time_arr_unpad = list(time_arr_pad[pca_mask])
        
        for i in range(len(flux_arr_padded)):
            flux_arr_unpad.append(flux_arr_padded[i][padding_mask[i]])
            time_arr_unpad.append(time_arr_padded[i][padding_mask[i]])
    
        return flux_arr_unpad,time_arr_unpad


def store_PCAed_flux_into_light_curve_dataframe(name_of_flux, list_light_curve, flux_arr_unpad):
    """
    Store PCA-ed flux array into list of light curve DataFrames.
    arguments
    =========
    name_of_flux: str-like
        Name of the flux column used to PCA.
        PCA-ed flux is saved to column of "pca_" + name_of_flux.
    list_light_curve: 
    flux_arr_unpad:
    returns
    =======
    list_light_curve: list of DataFrame
        list_light_curve where the column of ["pca_" + name_of_flux] is updated
    """
    column_saved = "pca_" + str(name_of_flux)
    for i, df_light_curve in enumerate(list_light_curve):
        df_light_curve[column_saved] = flux_arr_unpad[i]
    return list_light_curve


def output_PCAed_light_curve_to_hdf5(dir_output, source_ids, list_light_curve, date):
    """
    Output HDF5 files containing light curve data with PCA-ed flux of each star.

    Names of the HDF5 files are:

        $(dir_output)/light_curve_ + "source_id" + ".hdf5"

    arguments
    =========
    dir_output : str
        path of directory where light curves of stars are written
    source_ids : list of str
        list of source IDs of stars.
        catalog_name + source_id = e.g. "Gaia-DR2_0000000000".
    list_light_curve : 
    date : int or str
        observed date
    """

    os.makedirs(dir_output, exist_ok=True)

    for i, source_id in enumerate(source_ids):
        fpath = os.path.join(dir_output, "light_curve_" + str(source_id) + ".hdf5")
        try:
            list_light_curve[i].to_hdf(fpath, mode="a", key=str(date), format="table")
        except Exception as e:
            raise Exception(e)
            continue

    return

def main():
    """
    Main Function
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    run_start_time = prepare_run.get_start_time_str()
    logger = prepare_run.prepare_logger_merge(os.path.join(directory_log_pca, date_obs), run_start_time )
    logger.info("Start processing\n")

    if rank == 0:
        #ある観測日に撮られた星の情報を取得。read_output.py、catalog_star_history.py、util.pyが必要
        try:
            df_catalog = read_output.read_catalog_star_header_hdf5(dir_catalog_star+"catalog_stars_" + str(date_obs) + ".hdf5")
        except:
            logger.info("Catalog not found")
            return 
        logger.info("Finish reading catalog stars HDF5 file")
        df_catalog_all_stars, list_frame_history_of_star = catalog_star_history.construct_frame_history_of_star(df_catalog)
        logger.info("Finish construct frame history of stars")
        
        #観測履歴で星をgrouping
        dict_of_obs_history_index = group_stars_based_on_obs_history(df_catalog_all_stars.index,list_frame_history_of_star)
        logger.info("Finish grouping frame history of stars")
        
        #n_mumber_star_min (=10: デフォルト)より所属星の数が少ないグールプと多いグループに分ける
        pca_group_dict_index, nonpca_group_dict_index = divide_group_into_pca_and_nonpca(dict_of_obs_history_index, n_member_star_min=n_member_star_min)

        store_group_ids_to_df_catalog_all_stars(df_catalog_all_stars, pca_group_dict_index, nonpca_group_dict_index)
        path_output = os.path.join(directory_log_pca, date_obs, run_start_time, "df_catalog_all_stars.hdf5")
        df_catalog_all_stars.to_hdf(path_output, key="data")
        print(pca_group_dict_index, nonpca_group_dict_index )
        pca_group_dict    = replace_index_to_star_name(df_catalog_all_stars, pca_group_dict_index)
        nonpca_group_dict = replace_index_to_star_name(df_catalog_all_stars, nonpca_group_dict_index)

        logger.info("Total # of stars:      {:10d}".format(df_catalog_all_stars.shape[0]))
        Nstars_PCAed     = int(np.sum([len(val) for val in pca_group_dict.values()]))
        Nstars_not_PCAed = int(np.sum([len(val) for val in nonpca_group_dict.values()]))
        logger.info("# of stars PCA-ed:     {:10d}".format(Nstars_PCAed))
        logger.info("# of stars not PCA-ed: {:10d}".format(Nstars_not_PCAed ))
        del(df_catalog, list_frame_history_of_star, df_catalog_all_stars, dict_of_obs_history_index, pca_group_dict_index, nonpca_group_dict_index)

    else:
        pca_group_dict = None

    pca_group_dict = comm.bcast(pca_group_dict, root=0)

    # assign subgroup of pca_group_dict to each MPI process
    task_weights = [len(val) for val in pca_group_dict.values()]
    array_rank_assign = manage_job.assign_jobs_to_each_mpi_process_merge(task_weights, size)
    pca_group_dict = dict(zip(
        np.array(list(pca_group_dict.keys()  ))[array_rank_assign == rank].tolist(),
        np.array(list(pca_group_dict.values()))[array_rank_assign == rank].tolist(),
        ))

    #各PCAグループごと、各fluxの種類ごとにPCA 
    group_keys = list(pca_group_dict.keys())
    if len(group_keys) == 0:
        logger.info("No PCA-group of stars for this MPI process\nFinish all processes")
        return

    # DEBUG
    N_group_keys = len(group_keys)
    for i, group_key in enumerate(group_keys):
    #  for group_key in group_keys[:1]:
        if (i%1 == 0):
            logger.info("Processing group i={:9d} / {:9d}".format(i, N_group_keys))

        source_ids = pca_group_dict[group_key]
        source_ids, list_light_curve = load_light_curves(source_ids, dir_light_curve, date_obs)

        flux_keys = list_light_curve[0].keys()
        name_of_fluxes = [x for x in flux_keys if (x.startswith('flux_') or x.startswith('cflux'))*(not x.endswith('_err'))]

        # DEBUG
        for name_of_flux in name_of_fluxes:
        #  for name_of_flux in name_of_fluxes[:1]:
            source_ids,flux_arr_unpad,time_arr_unpad = pca_for_a_group_and_a_fluxtype(name_of_flux, list_light_curve, source_ids, n_pca_component, filepath_random, index_random_percentile, noise_amp_arr, thr_per_correlation, thr_per_variability)
            list_light_curve = store_PCAed_flux_into_light_curve_dataframe(name_of_flux, list_light_curve, flux_arr_unpad)

        output_PCAed_light_curve_to_hdf5(directory_light_curve_out_pca, source_ids, list_light_curve, date_obs)

    logger.info("Finish all processes")
    return

if __name__ == "__main__":
    main()

