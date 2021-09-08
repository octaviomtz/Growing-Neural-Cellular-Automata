from lib.utils_lung_segmentation import square


def test_square(value_4):
    square_result = square(value_4)
    assert square_result == 16