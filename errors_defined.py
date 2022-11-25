"""
Error classes defined manually

Author: Kojiro Kawana
"""

class FitsReadError(Exception):
    """Error while reading FITS file"""
    pass

class HDF5WriteError(Exception):
    """Error while writing HDF5 file"""
    pass

class SourceDetectionError(Exception):
    """Error while detection sources in FITS frame"""
    pass


