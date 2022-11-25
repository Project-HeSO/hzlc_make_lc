"""
Utilities to calc (ra, dec, error, and so on) for detected sources.

Author: Kojiro Kawana
"""
import os
import copy
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units
import sep

import utils

class DetectedSources(pd.DataFrame):
    """
    Class for soureces detected by sep.
    Inherit pandas.DataFrame as output of ``sep.extract``, with additional methods as follows:

    * ``calc_ra_dec_for_detected_objects``
    * ``calc_error_for_detected_objects_global_error``
    * ``output_to_hdf5``

    Todo
    ====
    Add following methods.
    * cross_match: detected sources => catalog sources (note that astropy.cross_match is not symmetric, and the current cross match is catalog sources => detected sources)
    """

    def aperture_photometry_circle(self, data_background_subtracted, background_error, r, inplace : bool = True, **kwargs_sep):
        """
        Perform circular aperture photometry. This function wraps ``sep.sum_circle``.
        This function does not take into account circular annulus.
        Aperture centers are taken to the centroids of the detected sources.

        arguments
        =========
        data_background_subtracted : numpy array
            FITS data where background is subtracted
        background_error           : numpy array or float
            Background error. Local RMS or global RMS.
        r : numpy array or float
            Radius of circular aperture in units of pixel.
        inplace : bool
            If True, write results to self. If False return results as dict.
        **kwargs : dict
            Keyword arguments passed to ``sep.sum_circle``.
            Keys of ["mask", "maskthresh", "bkgann", "gain", "subpix"] are passed.

        Returns
        =======
        results : dict, optional
            Returned if inplace is False. pandas.DataFrame-like structure.
            If argument ``r`` is not list, nor array, keys are ["flux_detected_aperture_r{:r}", "flux_detected_aperture_r{:r}_err", "flag_detected_aperture_r{:r}"].
            Else, keys are ["flux_detected_aperture_circle", "flux_detected_aperture_circle_err", "flag_detected_aperture_circle"]
        """

        results = {}
        if utils.is_array_like(r):
            keys = ["flux_detected_aperture_circle", "flux_detected_aperture_circle_err", "flag_detected_aperture_circle"]
        else:
            keys = ["flux_detected_aperture_r" + str(r), "flux_detected_aperture_r" + str(r) + "_err", "flag_detected_aperture_r" + str(r)]

        keys_requested_by_sep_sum_circle = ["mask", "maskthresh", "bkgann", "gain", "subpix"]
        kwargs = utils.get_slice_of_dictionay(kwargs_sep, keys_requested_by_sep_sum_circle)

        results[keys[0]], results[keys[1]], results[keys[2]] = sep.sum_circle(data_background_subtracted, self["x_centroid"], self["y_centroid"], r=r, err=background_error, **kwargs)
        if inplace:
            for k, v in results.items():
                self[k] = v
            return
        else:
            return results


    def calc_ra_dec_for_detected_objects(self, fits_header, inplace : bool = True):
        """
        calc ra, dec for detect sources with wcs and (x_catalog_position, y_catalog_position) coordinates.
    
        arguments
        =========
        fits_header: dict
            Dictionary containing information of fits header
        inplace: bool, default True
            Modify the DataFrame in place (do not create a new object).
    
        return
        ======
        df_objects : pandas.DataFrame, optional
            DataFrame of detected sources with (ra, dec). Returned if ``inplace==True``.
        """
    
        wcs_info = fits_header["wcs_info"]
        arr_ra_dec = np.array(wcs_info.all_pix2world(self["x_centroid"], self["y_centroid"], 0))
        if inplace:
            self["ra_centroid"]  = arr_ra_dec[0]
            self["dec_centroid"] = arr_ra_dec[1]
            #  self = pd.concat([self, pd.DataFrame(np.array(wcs_info.all_pix2world(self.x, self.y, 0)).T, columns=["ra", "dec"], index=self.index)], axis=1)
            return self
        else:
            df_objects = pd.concat([self, pd.DataFrame(arr_ra_dec.T, columns=["ra_centroid", "dec_centroid"], index=self.index)], axis=1)
            return df_objects
    
    def calc_error_for_detected_objects_global_error(self, fits_header, bkg_error, inplace : bool = True):
        """
        Calc ra, dec for detect sources with wcs and (x_catalog_position, y_catalog_position) coordinates.
    
        arguments:
            fits_header (dict)              : Dictionary containing information of fits header
            bkg_error   (float)             : Error of background. Now we assume that error is uniform over the image
            inplace     (bool, default True): Modify the DataFrame in place (do not create a new object).
    
        returns
        =======
        df_objects : pandas.DataFrame, optional
            DataFrame of detected sources with (flux_iso_err, cflux_err). Returned if ``inplace==True``.
        """
    
        err_squared = bkg_error * bkg_error
        flux_iso_err  = np.sqrt(self["npix"] * err_squared + self["flux_iso"]  / fits_header["GAIN"])
        cflux_err     = np.sqrt(self["npix"] * err_squared + self["cflux"] / fits_header["GAIN"])
        if inplace:
            self["flux_iso_err"]  = flux_iso_err
            self["cflux_err"] = cflux_err
            return
        else:
            df_objects = copy.deepcopy(self)
            df_objects["flux_iso_err"]  = flux_err  
            df_objects["cflux_err"] = cflux_err 
            return df_objects
    
    #  def calc_error_for_detected_objects_local_error(self, fits_header, bkg_sep, segmap):
    #     """
    #     calc ra, dec for detect sources with wcs and (x_catalog_position, y_catalog_position) coordinates.
    #
    #     arguments
    #     =========
    #     df_objects : DataFrame of detected sources
    #     fits_header: Dictionary containing information of fits header
    #     bkg_sep    : sep.Background
    #     segmap     : 2nd return of sep.Background with kwargs_sep_bkg["segmentation_map"] ==True
    #
    #     return
    #     ======
    #     df_objects : DataFrame of detected sources with (flux_err, cflux_err)
    #     """
    #     def calc_flux_err_for_local_bkg_err(df_sep_:pd.DataFrame, err2_, segmap_):
    #         N_source = df_sep_.shape[0]
    #         err_bkg = np.zeros(N_source, dtype=np.float64)
    #         for i in range(N_source):
    #             err_bkg[i] = _err2[np.where(segmap_ == i + 1)].sum()
    #         return err_bkg
    #
    #         #         _err2 = bkg_sep.back()**2
    #     #         err_bkg = np.zeros(df_sep.shape[0], dtype=np.float64)
    #     #         for i in range(df_sep.shape[0]):
    #     #             err_bkg[i] = _err2[np.where(segmap == i + 1)].sum()
    #     #
    #
    #             tstart = time.time()
    #            _err2 = bkg_sep.back()**2
    #            _err2_bkg = calc_flux_err_for_local_bkg_err(df_sep, _err2, segmap)
    #            df_sep = pd.concat([df_sep,
    #                                pd.DataFrame(np.array(
    #                                    [np.sqrt(_err2_bkg + df_sep.flux / gain),
    #                                     np.sqrt(_err2_bkg + df_sep.cflux / gain)]
    #                                ).T,
    #                                             columns=["flux_err", "cflux_err"], index=df_sep.index)],
    #                               axis=1)
    #     return self
    
    def to_pandas(self):
        """
        Convert to pandas.DataFrame

        Return
        ======
        df : pandas.DataFrame
        """
        return pd.DataFrame(self)
    
    def output_to_hdf5(self, dir_output, fits_header, time_index):
        """
        Output a HDF5 file containing detectes sources information.
    
        Names of the HDF5 file is:
    
            $(dir_output)/$(DATE-OBS)/light_curve/detected_sources_$(FRAME_ID).hdf5
    
            e.g. ``detected_sources/20200401/light_curve/detected_sources_TMQ1202004010000000101.hdf5``
    
        A HDF5 file has the following structure:
    
        * ``/data`` : pandas.DataFrame table with index of time_index
    
        arguments
        =========
        fits_header : dict
            dictionary containing all the information of fits header
        dir_output  : str
            path of directory where light curves of stars are written
        time_index  : int
            index of current time slice of FITS movie
        """
    
        self["time_index"] = time_index
        self = self.set_index("time_index", append=True).swaplevel()

        output_path = os.path.join(dir_output, fits_header["DATE-OBS"].replace("-", ""), "light_curve", "detected_sources_{:}.hdf5".format(fits_header["FRAME_ID"]))
        self.to_hdf(output_path, mode="a", key="data", format="table", append=True)

        return


def cross_match(df_catalog:pd.DataFrame, df_objects):
    """
    Cross match between catalog stars and detected objects.
    This function wraps ``astropy.SkyCoord: match_to_catalog_sky``.

    arguments
    =========
    df_catalog : pandas.DataFrame
        DataFrame of catalog stars
    df_objects : DetectedSources
        DataFrame of detected sources.
        Columns ["id_detected_source", "separation_cross_match"] are added.

    return
    ======
    results : dict
        Keys are ["id_detected_source", "separation_cross_match"].

    Todo
    ====
    Use options in match_to_catalog_sky: proper motions, obstime, and so on.
    """
    coords_catalog = SkyCoord(
            df_catalog["ra" ].values * astropy.units.deg,
            df_catalog["dec"].values * astropy.units.deg)
    coords_sep = SkyCoord(
            df_objects["ra_centroid" ].values * astropy.units.deg,
            df_objects["dec_centroid"].values * astropy.units.deg)

    results = {}
    results["id_detected_source"], results["separation_cross_match"], __ = coords_catalog.match_to_catalog_sky(coords_sep)

    return results

