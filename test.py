import cv2
import numpy as np
from brightest_quadrilateral import find_area, find_max_patches
import unittest


class BrightestQuadrilateralTest(unittest.TestCase):
    # Test case : There exist a convex solution
    def test_find_area_convex(self):
        corners = [(0, 0), (0, 100), (100, 0), (100, 100)]
        expected_area = 10000
        result_area, result_order = find_area(corners)
        self.assertEqual(result_area, expected_area)

    # Test case :  Quadrilateral - non convex
    def test_find_area_non_convex(self):
        corners = [(0, 0), (0, 100), (100, 0), (30, 30)]
        expected_areas = [3000, 4500]
        result_area, result_order = find_area(corners)
        self.assertIn(result_area, expected_areas)

    # Test case : Triangle
    def test_find_area_triangle(self):
        corners = [(0, 10), (0, 30), (0, 80), (100, 30)]
        expected_area = 3500
        result_area, result_order = find_area(corners)
        self.assertEqual(result_area, expected_area)

    # Test case : Line
    def test_find_area_line(self):
        corners = [(0, 0), (0, 4), (0, 6), (0, 8)]
        expected_area = 0
        result_area, result_order = find_area(corners)
        self.assertEqual(result_area, expected_area)

    # Test case: Line - not parallel to image axes
    def test_find_area_none_axe_line(self):
        corners = [(0, 0), (2, 2), (4, 4), (8, 8)]
        expected_area = 0
        result_area, result_order = find_area(corners)
        self.assertEqual(result_area, expected_area)

    # Test case 4 separate patches
    def test_brightest_patches_scattered(self):

        image = np.zeros((300, 300), dtype=np.uint8)

        # Set the 3x3 patches at specified positions
        image[10:13, 10:13] = 255
        image[100:103, 100:103] = 200
        image[250:253, 250:253] = 150
        image[50:53, 50:53] = 50

        num_patches = 4
        patch_size = [3, 3]

        # Expected output
        expected_result = [(11, 11), (101, 101), (251, 251), (51, 51)]

        # Call the function
        result = find_max_patches(image, num_patches, patch_size)
        self.assertEqual(result, expected_result)

    # Test case 2: Overlapping patches
    def test_brightest_patches_overlapping(self):
        image = np.zeros((300, 300), dtype=np.uint8)

        # Set the 25x25 white block at position (10, 10)
        image[10:35, 10:35] = 255

        num_patches = 4
        patch_size = [5, 5]

        # Expected output
        expected_result = [(12, 12), (12, 17), (12, 22), (12, 27)]

        # Call the function
        result = find_max_patches(image, num_patches, patch_size)
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
