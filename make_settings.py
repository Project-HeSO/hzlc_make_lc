import sys
import os
import numpy as np


file_out = open("path_info.py","w")
alpsdir = "/alps2"
product_dir = "/blacksmith"
catalog_dir = "/blacksmith"
catalog_name = "ohsawa_renewed_20210429.pickle"
number_fits_for_debug = 10
date_obs = sys.argv[1]

file_out.write('alpsdir ="%s"\n' % alpsdir)
file_out.write('product_dir ="%s"\n' % product_dir)
file_out.write('catalog_dir ="%s"\n' % catalog_dir)
file_out.write('catalog_name ="%s"\n' % catalog_name)
file_out.write('date_obs =%s\n' % date_obs)
file_out.write('number_fits_for_debug =%d\n' % number_fits_for_debug)

