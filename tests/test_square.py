from lib.utils_lung_segmentation import square


def test_square(value_4):
    subject = square(value_4)

    assert subject == 16