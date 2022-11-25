"""
Functions to process FITS data with sep

Author: Kojiro Kawana
"""

import numpy as np
import pandas as pd
import sep
from errors_defined import *
import utils
import utils_for_detected_sources

def background_subtraction_sep(fits_snapshot, **kwargs_sep_bkg):
    """
    Estimate background and subtruct the local background from the data of FITS snapshot with sep/SExtractor.
    This function wraps sep.Background(data, mask=None, maskthresh=0.0, bw=64, bh=64, fw=3, fh=3, fthresh=0.0).

    arguments
    =========
    fits_snapshot : numpy.array(float)
        2D FITS data. same as hdul[0].data[i]
    **kwargs_sep_bkg: dict
        keyword arguments passed to sep.Background()

    return
    ======
    bkg_sep            : sep.Background
    data_bkg_subtracted: numpy.array(float)
        fits_snapshot - background.back()

    references
    ==========
    https://photutils.readthedocs.io/en/stable/background.html

    Todo
    ====
    Optimize parameters of ``kwargs_sep_bkg``
    """
    bkg_sep = sep.Background(fits_snapshot.byteswap().newbyteorder(), **kwargs_sep_bkg)
    data_bkg_subtracted = fits_snapshot - bkg_sep.back()
    return bkg_sep, data_bkg_subtracted


def source_detection_and_photometry_sep(fits_snapshot, background_error, fits_header, thresh, segmentation_map=False, N_detection_max=300000, multiply_next=8, **kwargs_sep_detection):
    """
    Detect objects in a FITS snapshot with the background previously estimated and calculate FLUX_ISO for the detected objects.
    If more than 3e5 sources are detected, try to increase the N_MAX_SOURCES upto N_detection_max.
    The algorithm is based on sep/SExtractor.

    This function wraps 

        sep.extract(data, thresh, err=None, mask=None, minarea=5,\
               filter_kernel=default_kernel, filter_type='matched',\
               deblend_nthresh=32, deblend_cont=0.005, clean=True,\
               clean_param=1.0, segmentation_map=False)

    arguments
    =========
    fits_snapshot        : numpy.array(float)
        2D FITS data. same as hdul[0].data[i] - sep.Background
    background_error     : (float or array_like)
        sep.Background.globalrms or sep.Background.rms() (local background error)
    fits_header          : dict
        Dictionary containing information of fits header
    thresh               : float
        threshold to detect sources. argument passed to sep.detection(). sep default is 1.5.
    segmentation_map     : bool
        If True, also return a “segmentation map” giving the member pixels of each object. Default is False.
    N_detection_max      : int
        maximum number of sources allowed to be detected. sep default is 3e5.
    multiply_next        : int
        if number of detected sources exceeds limit, increase the limit by *= multiply_next
    kwargs_sep_detection : dict
        keyword arguments passed to sep.detection()

    returns
    =======
    detected_sources: ``DetectedSources``
        Extracted object parameters. ``DetectedSources`` is inheeitance of pandas.DataFrame with additional methods. Available fields are:

        * thresh (float) Threshold at object location.
        * npix (int) Number of pixels belonging to the object.
        * tnpix (int) Number of pixels above threshold (unconvolved data).
        * xmin, xmax (int) Minimum, maximum x coordinates of pixels.
        * ymin, ymax (int) Minimum, maximum y coordinates of pixels.
        * x_centroid, y_centroid (float) object barycenter (first moments).
        * x2, y2, xy (float) Second moments.
        * errx2, erry2, errxy (float) Second moment errors. Note that these will be zero if error is not given.
        * a, b, theta (float) Ellipse parameters, scaled as described by Section 8.4.2 in “The Source Extractor Guide” or Section 10.1.5-6 of v2.13 of SExtractor’s User Manual.
        * cxx, cyy, cxy (float) Alternative ellipse parameters.
        * cflux (float) Sum of member pixels in convolved data.
        * flux_iso (float) Sum of member pixels in unconvolved data.
        * cpeak (float) Peak value in convolved data.
        * peak (float) Peak value in unconvolved data.
        * xcpeak, ycpeak (int) Coordinate of convolved peak pixel.
        * xpeak, ypeak (int) Coordinate of unconvolved peak pixel.
        * flag_sep (int) Extraction flags.

    segmap : numpy.array(int), optional
        Array of integers with same shape as data. Pixels not belonging to any object have value 0. All pixels belonging to the i-th object (e.g., objects[i]) have value i+1. Only returned if segmentation_map=True.


    Note
    ====
    * Now we use bkg_sep.global_rms because it is much faster. Using local rms enables more accurate photometry but is slower.
    * detected_sources["theta"] should be between -pi/2 and pi/2, but sometimes this is not satisfied due to floating point precision. Thus if this is not satisfied, we add +- pi in order to satisfy it.

    Todo
    ====
    Optimize parameters of ``kwargs_sep_detection``
"""

    N_detection_max_now = int(3e5)
    sep.set_extract_pixstack(N_detection_max_now)
    while (N_detection_max_now < N_detection_max):
        try:
            if segmentation_map == True:
                objects, segmap = sep.extract(fits_snapshot, thresh = thresh, err = background_error, gain=fits_header["GAIN"], segmentation_map=segmentation_map, **kwargs_sep_detection)
            else:
                objects         = sep.extract(fits_snapshot, thresh = thresh, err = background_error, gain=fits_header["GAIN"], segmentation_map=segmentation_map, **kwargs_sep_detection)
            break
        except Exception as e:
            # Increase the max number of detected sources and try re-detect
            N_detection_max_now *= multiply_next
            sep.set_extract_pixstack(N_detection_max_now)
            if (N_detection_max_now < N_detection_max):
                continue
            else:
                raise SourceDetectionError("more than N_detection_max={:d} sources are detected".format(N_detection_max))
                break

    df_objects = pd.DataFrame(objects)
    df_objects.loc[(df_objects["theta"] - 0.5 * np.pi) > 0.0, "theta"] -= 0.5 * np.pi
    df_objects.loc[(df_objects["theta"] + 0.5 * np.pi) < 0.0, "theta"] += 0.5 * np.pi
    dict_rename_columns = {
        "flux" : "flux_iso",
        "flag" : "flag_sep",
        "x"    : "x_centroid",
        "y"    : "y_centroid"
    }
    df_objects.rename(columns=dict_rename_columns, inplace=True)
    detected_sources = utils_for_detected_sources.DetectedSources(df_objects)


    if segmentation_map == True:
        return detected_sources, segmap
    else:
        return detected_sources 



def aperture_photometry_auto(data_background_subtracted, background_error, objects, fits_header, xy_centers=None, axes_multiply=6.0, k_kron_radius=2.5, minimum_diameter=3.5, inplace=True, **kwargs):
    """
    Perform automatic aperture photometry implemented in SExtractor (FLUX_AUTO) for detected sources with sep.

    Args:
        data_background_subtracted (numpy array)                   : FITS data where background is subtracted
        background_error           (numpy array or float)          : Background error. local RMS or global RMS.
        objects                    (dict/PyTables/pandas.DataFrame): return of sep.extract or source_detection_and_photometry_sep
        xy_centers                 (dict/PyTables/pandas.DataFrame): Defaults None. If given, use centers of the aperutes as xy_centers, instead of objects["x", "y"]
        axes_multiply              (numpy array or float)          : Muliply factor used to estimate Kron radius. Default is 6.0 (from SExtractor)
        k_kron_radius              (numpy array or float)          : Muliply factor used to estimate FLUX_AUTO. Default is 2.5 (from SExtractor)
        inplace                    (bool)                          : If True, write results to ``objects``. If False return results as dict
        kwargs                     (dict)                          : keyword arguments for sep.* Must include ``axes_multiply`` and ``k_kron_radius``

    Returns
    =======
    results :dict, optional
        Returned if inplace is False. pandas.DataFrame-like structure

    Note
    ====
    * Background error estimation from annulus can be enabled by ``bkgann`` option in sep.sum_*().
    * See SExtractor document for the choice of the default parameter.

    Todo
    ====
    Add option for ``k_kron_radius`` in sep.sum_ellipse() to vary by the background error and the source flux (in order to maximize S/N).
    """

    if xy_centers is None:
        x = objects["x_centroid"]
        y = objects["y_centroid"]
    else:
        x = xy_centers["x_centroid"]
        y = xy_centers["y_centroid"]

    kwargs_requested_by_sep_kron_radius = ["mask", "maskthresh"]
    kwargs_requested_by_sep_sum = ["mask", "maskthresh", "bkgann", "subpix"]
    kwargs_for_kron_radius = utils.get_slice_of_dictionay(kwargs, kwargs_requested_by_sep_kron_radius)
    kwargs_for_sep_sum = utils.get_slice_of_dictionay(kwargs, kwargs_requested_by_sep_sum)


    kron_radius, flag_kron = sep.kron_radius(data_background_subtracted, x, y, objects["a"], objects["b"], objects["theta"], r=axes_multiply, **kwargs_for_kron_radius)
    flux, flux_error, flag = sep.sum_ellipse(data_background_subtracted, x, y, objects["a"], objects["b"], objects["theta"], r=k_kron_radius * kron_radius, err=background_error, gain=fits_header["GAIN"], **kwargs_for_sep_sum )

    if (minimum_diameter is not None) and (minimum_diameter > 0):
        r_min = 0.5 * minimum_diameter
        use_circle = kron_radius * np.sqrt(objects["a"] * objects["b"]) < r_min
        cflux, cfluxerr, cflag = sep.sum_circle(data_background_subtracted, x[use_circle], y[use_circle], r_min, gain=fits_header["GAIN"], **kwargs_for_sep_sum)
        flux[use_circle]    = cflux
        flux_error[use_circle] = cfluxerr
        flag[use_circle]    = cflag

    if inplace == True:
        objects["kron_radius"]    = kron_radius
        objects["flag_kron"]      = flag_kron
        objects["flux_auto"]      = flux
        objects["flux_auto_err"]  = flux_error
        objects["flag_flux_auto"] = flag
        return
    else:
        results = {}
        results["kron_radius"]    = kron_radius
        results["flag_kron"]      = flag_kron
        results["flux_auto"]      = flux
        results["flux_auto_err"]  = flux_error
        results["flag_flux_auto"] = flag
        return results



def aperture_photometry_circle_catalog_stars(data_background_subtracted, background_error, df_catalog, r, inplace=False,**kwargs_sep):
    """
    Perform circular aperture photometry for catalog stars. This function wraps ``sep.sum_circle``.
    This function does not take into account circular annulus.
    Aperture centers are calculated by WCS and (RA, Dec) of catalog values.

    arguments
    =========
    data_background_subtracted : numpy array
        FITS data where background is subtracted
    background_error           : numpy array or float
        Background error. Local RMS or global RMS.
    df_catalog       : pandas.DataFrame
        DataFrame of catalog stars. ["x_catalog_position"], ["y_catalog_position"] columns are used as aperture centers.
    r : numpy array or float
        Radius of circular aperture in units of pixel.
    inplace: bool
        If True, write results to df_catalog. If False, return resutls as dict
    **kwargs : dict
        Keyword arguments passed to ``sep.sum_circle``.
        Keys of ["mask", "maskthresh", "bkgann", "gain", "subpix"] are passed.

    return
    ======
    results : dict, optional
        Returned if inplace==False.
        Keys are ["flux_detected_aperture_r{:r}", "flux_detected_aperture_r{:r}_err", "flag_detected_aperture_r{:r}"].
        Values are numpy array of (float, float, int16)
    """

    keys = ["flux_catalog_aperture_r" + str(r), "flux_catalog_aperture_r" + str(r) + "_err", "flag_catalog_aperture_r" + str(r)]

    keys_requested_by_sep_sum_circle = ["mask", "maskthresh", "bkgann", "gain", "subpix"]
    kwargs = utils.get_slice_of_dictionay(kwargs_sep, keys_requested_by_sep_sum_circle)

    x = df_catalog["x_catalog_position"].to_numpy()
    y = df_catalog["y_catalog_position"].to_numpy()

    if inplace:
        df_catalog[keys[0]], df_catalog[keys[1]], df_catalog[keys[2]] = sep.sum_circle(data_background_subtracted, x, y, r=r, err=background_error, **kwargs)
        return
    else:
        results = {}
        results[keys[0]],    results[keys[1]],    results[keys[2]]    = sep.sum_circle(data_background_subtracted, x, y, r=r, err=background_error, **kwargs)
        return results

