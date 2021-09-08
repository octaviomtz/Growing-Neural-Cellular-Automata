from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.morphology import disk, binary_closing
from skimage.filters import roberts
from scipy.ndimage import binary_fill_holes
from lib.utils_superpixels import coords_min_max_2D
from collections import namedtuple
from operator import mul
from functools import reduce
import numpy as np

def get_segmented_lungs(im, thresh=.5):
    """This funtion segments the lungs from the given 2D slice.
    https://www.kaggle.com/malr87/lung-segmentation

    Args:
        im (2D numpy array): single lung slice
        thresh (float, optional): [description]. Defaults to .5.

    Returns:
        numpy array: segmented lungs
    """
    # Convert into a binary image. 
    binary = im < thresh # thresh=604
    
    # Remove the blobs connected to the border of the image
    cleared = clear_border(binary)

    # Label the image
    label_image = label(cleared)

    # Keep the labels with 2 largest areas
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    for region in regionprops(label_image):
        # print (region.area, areas[-2])
        if region.area < areas[-2]:
            for coordinates in region.coords:                
                label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0

    # Closure operation with disk of radius 12
    selem = disk(10)
    binary = binary_closing(binary, selem)
    
    # Fill in the small holes inside the lungs
    edges = roberts(binary)
    binary = binary_fill_holes(edges)

    # Superimpose the mask on the input image
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    
    return im



Info = namedtuple('Info', 'start height')
def max_rectangle_size(histogram):
    """Find height, width of the largest rectangle that fits entirely under
    the histogram.
    # https://stackoverflow.com/questions/2478447/find-largest-rectangle-containing-only-zeros-in-an-n%C3%97n-binary-matrix
    """
    stack = []
    top = lambda: stack[-1]
    max_size = (0, 0) # height, width of the largest rectangle
    pos = 0 # current position in the histogram
    for pos, height in enumerate(histogram):
        start = pos # position where rectangle starts
        while True:
            if not stack or height > top().height:
                stack.append(Info(start, height)) # push
            elif stack and height < top().height:
                max_size = max(max_size, (top().height, (pos - top().start)),
                               key=area)
                start, _ = stack.pop()
                continue
            break # height == top().height goes here

    pos += 1
    for start, height in stack:
        max_size = max(max_size, (height, (pos - start)), key=area) 
        # print(max_size, height, (pos - start))
    return max_size, pos, start, height

def get_max_rect_in_polygon(mat, value=0):
    """Find height, width of the largest rectangle containing all `value`'s."""
    it = iter(mat)
    hist = [(el==value) for el in next(it, [])]
    max_size, pos, start, height = max_rectangle_size(hist)
    for idx_row, row in enumerate(it):
        hist = [(1+h) if el == value else 0 for h, el in zip(hist, row)]
        old_max_size = max_size
        max_size = max(max_size, max_rectangle_size(hist)[0], key=area)
        new_size = max(max_size, max_rectangle_size(hist)[0], key=area)
        if new_size > old_max_size:
            idx_row_max = idx_row
            hist_max = hist
            row_max = row
            pos_max = pos 
            start_max = start

    HEIGHT, WIDTH = max_size
    X2 = np.where(hist_max==np.max(hist_max))[0][0]
    Y2 = idx_row_max
    X1 =  int(np.min(np.where(np.array(hist_max)>=HEIGHT)))
            
    # return max_size, pos, start, height, idx_row_max, hist_max, row_max
    return HEIGHT, WIDTH, Y2, X1, X2, hist_max

def area(size):
    return reduce(mul, size)

def square(x):
    return x ** 2