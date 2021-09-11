from lib.utils_lung_segmentation import square


def test_square():
    subject = square(4)

    assert subject == 16