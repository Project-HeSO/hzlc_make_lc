"""
Modules to read HDF5/npy file output by the ``hzlc`` module.

Author: Kojiro Kawana
"""

import h5py
import numpy as np
import pandas as pd

import utils


def read_movie_hdf5(path, date=None, read_header=False):
    """
    read data in a movie HDF5 file.

    arguments
    =========
    path: str
        path of hdf5 file containing data
    date: str/int or list of str/int (optional)
        Key (observed date) of the data. e.g. 20200401 or [20200401, 20200402].
        If None, read all the date.
    read_header: bool
        If True, read header info and return it. Defalut to False.

    returns
    =======
    dates : numpy.array(int)
        Observed dates. If date is not None, return the argument, date.
    dfs_utc : (list of) pandas.DataFrame
        List of dataframe of UTC.
        Length is number of dates observed or ``len(date)``.
    movie_data_catalog_position : (list of) numpy.array(float)
        List of 3d data of movie (time, x, y) with its center at the catalog RA, Dec position.
        Length is number of dates observed or ``len(date)``.
    movie_data_centroid : (list of) numpy.array(float)
        List of 3d data of movie (time, x, y) with its center at the detected source position.
        Length is number of dates observed or ``len(date)``.
    df_header : pandas.Series (optional)
        Header info recored in ["/header"].

    """

    dfs_utc                     = []
    movie_data_catalog_position = []
    movie_data_centroid         = []
    with h5py.File(path, mode="r") as hf:
        if date is None:
            dates = list(hf.keys())
            dates.remove("header")
        else:
            date_int = int(date)
            if utils.is_array_like(date_int):
                dates = np.array(date_int).astype(int)
            else:
                dates = np.array([date_int]).astype(int)

        for key in dates:
            movie_data_catalog_position.append(np.array(hf["{:}/movie_catalog_position".format(str(key))]))
            movie_data_centroid.append(np.array(hf["{:}/movie_centroid".format(str(key))]))

    for key in dates:
        dfs_utc.append(pd.read_hdf(path, key="{:}/utc".format(str(key))))

    if not read_header:
        return dates, dfs_utc, movie_data_catalog_position, movie_data_centroid
    else:
        df_header = pd.read_hdf(path, key="header")
        return dates, dfs_utc, movie_data_catalog_position, movie_data_centroid, df_header


def read_light_curve_hdf5(path, date=None, read_header=False):
    """
    read data in a light curve HDF5 file.

    arguments
    =========
    path: str
        path of hdf5 file containing data
    date: str/int or list of str/int (optional)
        Key (observed date) of the data. e.g. 20200401 or [20200401, 20200402].
        If None, read all the date.
    read_header: bool
        If True, read header info and return it. Defalut to False.

    returns
    =======
    dates : numpy.array(int)
        Observed dates. If date is not None, return the argument, date.
    dfs_light_cuve : (list of) pandas.DataFrame
        List of dataframe of light curve.
        Length is number of dates observed or ``len(date)``.
    df_header : pandas.Series (optional)
        Header info recored in ["/header"].

    """

    dfs_light_curve = []
    with h5py.File(path, mode="r") as hf:
        if date is None:
            dates = list(hf.keys())
            dates.remove("header")
        else:
            date_int = int(date)
            if utils.is_array_like(date_int):
                dates = np.array(date_int).astype(int)
            else:
                dates = np.array([date_int]).astype(int)

    for key in dates:
        dfs_light_curve.append(pd.read_hdf(path, key=(str(key))))
        
    if not read_header:
        return dates, dfs_light_curve
    else:
        df_header = pd.read_hdf(path, key="header")
        return dates, dfs_light_curve, df_header

def read_LC_hdf5(path, date=None, read_header=False):
    """
    read data in a light curve HDF5 file.

    arguments
    =========
    path: str
        path of hdf5 file containing data
    date: str/int or list of str/int (optional)
        Key (observed date) of the data. e.g. 20200401 or [20200401, 20200402].
        If None, read all the date.
    read_header: bool
        If True, read header info and return it. Defalut to False.

    returns
    =======
    dates : numpy.array(int)
        Observed dates. If date is not None, return the argument, date.
    dfs_light_cuve : (list of) pandas.DataFrame
        List of dataframe of light curve.
        Length is number of dates observed or ``len(date)``.
    df_header : pandas.Series (optional)
        Header info recored in ["/header"].

    """

    dfs_light_curve = []
    with h5py.File(path, mode="r") as hf:
        if date is None:
            dates = list(hf.keys())
            dates.remove("header")
        else:
            date_int = int(date)
            if utils.is_array_like(date_int):
                dates = np.array(date_int).astype(int)
            else:
                dates = np.array([date_int]).astype(int)

    for key in dates:
        try: 
            pd.read_hdf(path, key=(str(key)))
            dfs_light_curve.append(pd.read_hdf(path, key=(str(key))))
        except:
            pass 

    if not read_header:
        return dates, dfs_light_curve
    else:
        df_header = pd.read_hdf(path, key="header")
        return dates, dfs_light_curve, df_header

    
def read_catalog_star_header_hdf5(path):
    """
    read data in a HDF5 file containing catalog star information in a observed date.

    arguments
    =========
    path: str
        path of hdf5 file containing data. e.g. "fits_header/fits_headers_20200401.hdf5"

    returns
    =======
    df_catalog: pandas.DataFrame
        Catalog informaiton of the observed catalog stars in a certain date.
    """

    df_catalog = pd.read_hdf(path, key="data")
    return df_catalog

def read_fits_header_hdf5(path):
    """
    read data in a FITS header HDF5 file.

    arguments
    =========
    path: str
        path of hdf5 file containing data. e.g. "fits_header/fits_headers_20200401.hdf5"

    returns
    =======
    df_fits_header : pandas.DataFrame
        FITS header info of a certain date.
    """

    df_fits_header = pd.read_hdf(path, key="data")
    return df_fits_header

def read_detected_sources_hdf5(path):
    """
    read data in a detected sources HDF5 file.

    arguments
    =========
    path: str
        path of hdf5 file containing data. e.g. "detected_sources/20200401/light_curve/detected_sources_TMQ1202004010000000101.hdf5"

    returns
    =======
    df_detected_sources: pandas.DataFrame
        Light curves of detected sources.
    """
    df_detected_sources = pd.read_hdf(path, key="data")
    return df_detected_sources

def read_light_curve_npy(path):
    """
    read data in a light curve npy file.

    arguments
    =========
    path: str
        path of npy file containing data. e.g. "light_curve_npy/light_curve_TMQ1202004010000000101.npy"

    returns
    =======
    light_curves: numpy structured array
        See ``light_curve_container._make_dtype_light_curves``.
    """
    light_curves = np.load(path, allow_pickle=True)
    return light_curves

def read_light_curve_pickle(path):
    """
    read data in a light curve pickle file.

    arguments
    =========
    path: str
        path of npy file containing data. e.g. "light_curve_pickle/light_curve_TMQ1202004010000000101.pickle"

    returns
    =======
    df_light_curves: pandas.DataFrame
        Light curves of catalog stars.
    """
    df_light_curves = pd.read_pickle(path)
    return df_light_curves

def read_catalog_stars_pickle(path):
    """
    read data in a pickle file of catalog stars information.

    arguments
    =========
    path: str
        path of npy file containing data. e.g. "catalog_pickle/catalog_stars_TMQ1202004010000000101.pickle"

    returns
    =======
    df_catalog: pandas.DataFrame
        Header information of catalog stars.
    """
    df_catalog = pd.read_pickle(path)
    return df_catalog

def read_catalog_stars_hdf5(path):
    """
    read data in a HDF5 file of catalog stars information.

    arguments
    =========
    path: str
        path of npy file containing data. e.g. "catalog_star_header/catalog_stars_20200401.hdf5"

    returns
    =======
    df_catalog: pandas.DataFrame
        Header information of catalog stars.
    """
    df_catalog = pd.read_hdf(path, key="data")
    return df_catalog

def read_movie_detected_sources_npy(path):
    """
    read movie data in a detected sources npy file.

    arguments
    =========
    path: str
        path of npy file containing data. e.g. "detected_sources/20200401/movie/movie_detected_sources_TMQ1202004010000000101.hdf5"

    returns
    =======
    movie_data: numpy.array of numpy.array(float)
        len(movie_data) == FITS["NAXIS3"]. movie_data[i] is movies of detected sources in ``i`` time index.
    """
    movie_data = np.load(path, allow_pickle=True)
    return movie_data

def crossmatch_output_wth_anycatalog(df_output, df_any, id_cross = "source_id"):
    """
    crossmath output & input catalog

    arguments
    =========
    df_output: pandas.DataFrame
        catalog for output data
    df_any: pandas.DataFrame
        catalog (df_any) that has more columns than catalog for output (df_output) . By crossmatching df_output with df_any, we get more columns for observed targets
    id_cross: str
        column used for crossmatch
    returns
    =======
    new_catalog: pandas.DataFrame
        Crossmatched catalog
    """
            
    new_catalog = df_output.join(df_any.set_index(id_cross), rsuffix = "output_", on=id_cross)
    return new_catalog

def take_mdwarf_targets_from_outputcatalog_from_particular_date(dir_catalog_star, date,file_mdwarfcatalog="/blacksmith/catalog/used_now/ohsawa_+_mdwarf+wd.pickle"):
    """
    crossmath output & input catalog

    arguments
    =========
    dir_catalog_star: str
        directory that has catalog for outputs
    date: int
        date for observation
    file_mdwarfcatalog: str
        path to mdwarf catalog, which has columns of mass, GAIA ID
        
    returns
    =======
    df_catalog_target: pandas.DataFrame
        Crossmatched M dwarf catalog for observation on "date"
    """ 
    df_catalog = read_catalog_star_header_hdf5(dir_catalog_star+ "catalog_stars_" + str(date) + ".hdf5")
    mdwarf_catalog = pd.read_pickle(file_mdwarfcatalog)
    new_catalog = crossmatch_output_wth_anycatalog(df_catalog, mdwarf_catalog, id_cross = "source_id")
    df_catalog_target = new_catalog[pd.notnull(new_catalog["mass"])]
    df_catalog_target = df_catalog_target.loc[:,['catalog_name','source_id','ra','dec']].drop_duplicates()
    
    return df_catalog_target

