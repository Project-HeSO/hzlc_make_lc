"""
Handle header information of catalog-stars in FITS files.
Construct FITS frame history of stars, as a preparation for PCA.

Author: Kojiro Kawana
"""
import os
import numpy as np
import pandas as pd

import utils
import read_output

def merge_per_fits_frame_to_per_date(fpaths_read):
    """
    Read pickle files of catalog information of stars in FITS frames, and merge them to one DataFrame.

    argument
    ========
    fpaths_read : list of str
        File paths of pickle files. e.g. catalog_stars_TMQ....pickle

    return
    ======
    df_catalog_merged : pandas.DataFrame
        merged catalog DataFrame. ``frame_id`` column is added to the DataFrame read.
    """
    df_catalog_merged = []
    for fpath in fpaths_read:
        df_catalog = read_output.read_catalog_stars_pickle(fpath)
        df_catalog["frame_id"] = os.path.basename(fpath).split("_")[-1][:-7] # TMQ...
        df_catalog_merged.append(df_catalog)

    df_catalog_merged = pd.concat(df_catalog_merged)

    return df_catalog_merged

def drop_duplicate_in_catalog_stars(df_catalog_star, subset=None):
    """
    Extract unique sample of stars in a DataFrame.

    argument
    ========
    df_catalog_star: DataFrame
        DataFrame of header information of catalog-stars
    subset: list of str
        Columns used to judge whether duplicates. If None, use default values, ["catalog_name", "source_id"].

    return
    ======
    df_unique: pandas.DataFrame
        DataFrame where duplicated rows are removed. Sorted by the given ``subset``.
        The index is reset and the original index is moved to an adiitional column.
    """

    if subset is None:
        subset = ["catalog_name", "source_id"]

    df_catalog_star_reindexed = df_catalog_star.reset_index(drop=True)
    df_unique = df_catalog_star_reindexed[subset].drop_duplicates()
    df_unique = df_catalog_star_reindexed.loc[df_unique.index]
    df_unique.sort_values(subset, inplace=True, ignore_index=True)
    return df_unique


def construct_frame_history_of_star(df_catalog_star):
    """
    Construct FITS frame history of stars, as a preparation for PCA.

    argument
    ========
    df_catalog_star: DataFrame
        DataFrame of header information of catalog-stars

    return
    ======
    df_unique: pandas.DataFrame
        Same as the output of ``drop_duplicate_in_catalog_stars``.
        DataFrame where duplicated rows are removed. Sorted by the given ``subset``.
        The index is reset and the original index is moved to an adiitional column.
    list_frame_history_of_star : list of list of str
        Length = df_unique.shape[0].
        Each entry is history of a star, i.e. in which frames the star has been.

    Todo
    ====
    * Make faster with MPI-parallel?
        Now this process takes ~22 min for 20191005 data.
    """

    subset = ["catalog_name", "source_id"]

    df_unique = drop_duplicate_in_catalog_stars(df_catalog_star)
    frame_ids = np.unique(df_catalog_star["frame_id"])
    frame_ids_sorted, __ = utils.sort_frame_ids_by_time_order(frame_ids)

    list_frame_history_of_star = [[] for i in range(df_unique.shape[0])]
    for i, frame_id in enumerate(frame_ids_sorted):
        df_catalog = df_catalog_star[df_catalog_star["frame_id"] == frame_id]
        df_catalog_sorted = df_catalog.sort_values(subset).reset_index()
        star_ids = utils.extract_overlap_between_2_dataframes(df_unique, df_catalog_sorted, subset=subset, keep="last")

        for star_id in star_ids:
            list_frame_history_of_star[star_id].append(frame_id)

    return df_unique, list_frame_history_of_star
