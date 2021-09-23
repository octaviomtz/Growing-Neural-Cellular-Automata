import numpy as np
from skimage.measure import label   
from lib.utils_lung_segmentation import get_max_rect_in_mask


def getLargestCC(segmentation):
    '''find largest connected component
    return: binary mask of the largest connected component'''
    labels = label(segmentation)
    assert(labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def test_max_rect_in_mask(random_blobs):
    '''make sure that we can find the largest rectangle inside a bindary mask
    check that the coordinates of the rectangle are the coorect ones'''
    blobs_largest = getLargestCC(random_blobs)
    coords_largest =  get_max_rect_in_mask(blobs_largest)
    assert coords_largest == (83, 125, 143, 155)
