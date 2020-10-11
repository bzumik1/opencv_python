import cv2 as cv
import unittest
import numpy.testing as npt

from SoftwareVision import ImageBGR


class TestSoftwareVision(unittest.TestCase):
    def test_constructing_from_file(self):
        my_img = ImageBGR.from_file("resources/test_svao_small.jpg")
        img = cv.imread("resources/test_svao_small.jpg")

        npt.assert_array_equal(img, my_img.bgr())

    def test_raises_error_when_constructing_from_file_and_parameter_is_not_string(self):
        self.assertRaises(ValueError, ImageBGR.from_file, 1)

    def test_constructing_from_image(self):
        img = cv.imread("resources/test_svao_small.jpg")
        my_img = ImageBGR.from_array(img)

        npt.assert_array_equal(img, my_img.bgr())

    def test_raises_error_when_constructing_from_array_and_parameter_is_not_array(self):
        self.assertRaises(ValueError, ImageBGR.from_array, 1)

    def test_raises_error_when_resize_parameter_is_not_integer(self):
        my_img = ImageBGR.from_file("resources/test_svao_small.jpg")
        self.assertRaises(ValueError, my_img.resize, "a", 1)
        self.assertRaises(ValueError, my_img.resize, 1, "b")

    def test_raises_error_when_resize_parameter_is_smaller_than_zero(self):
        my_img = ImageBGR.from_file("resources/test_svao_small.jpg")
        self.assertRaises(ValueError, my_img.resize, 1, -1)
        self.assertRaises(ValueError, my_img.resize, -1, 1)



if __name__ == '__main__':
    unittest.main()
