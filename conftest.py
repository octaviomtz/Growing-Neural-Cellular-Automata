import pytest
import numpy as np
from skimage import filters

@pytest.fixture
def value_4():
    return 4

@pytest.fixture
def random_blobs():
    '''make random blobs to test mask_rect_in_mask according to
    https://scipy-lectures.org/packages/scikit-image/index.html'''
    n = 20
    l = 256
    im = np.zeros((l, l))
    np.random.seed(2)
    points = l * np.random.random((2, n ** 2))
    im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    im = filters.gaussian(im, sigma=l / (2. * n))
    blobs = im > im.mean()
    return blobs