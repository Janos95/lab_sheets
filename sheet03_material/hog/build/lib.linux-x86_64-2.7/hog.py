import hogpy
import numpy as np


def extract(im, nb_bins=9, cwidth=8, block_size=2, unsigned_dirs=True, clip_val=.2):
    # Your additional code goes here
    return hogpy.hog(im, nb_bins, cwidth, block_size, unsigned_dirs, clip_val)